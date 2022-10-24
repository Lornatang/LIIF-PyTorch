# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch import Tensor
from torch import nn
import numpy as np
from typing import Any
from utils import make_coord
from torch.nn import functional as F_torch

__all__ = [
    "liif_edsr"
]


class LIIF(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            encoder_channels: int = 64,
            out_channels: int = 3,
            encoder_arch: str = "edsr",
    ) -> None:
        super(LIIF, self).__init__()
        if encoder_arch == "edsr":
            self.encoder = _EDSR(in_channels, encoder_channels)
        else:
            self.encoder = _EDSR(in_channels, encoder_channels)

        mlp_channels = int(encoder_channels * 9)
        mlp_channels += 2  # Attach coord
        mlp_channels += 2  # Attach cell
        self.mlp = _MLP(mlp_channels, out_channels, [256, 256, 256, 256])

    def forward(self, x: Tensor, x_coord: Tensor, x_cell: Tensor = None) -> Tensor:
        return self._forward_impl(x, x_coord, x_cell)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor, x_coord: Tensor, x_cell: Tensor) -> Tensor:
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        features = self.encoder(x)

        features = F_torch.unfold(features, (3, 3), padding=(1, 1)).view(features.shape[0],
                                                                         features.shape[1] * 9,
                                                                         features.shape[2],
                                                                         features.shape[3])

        # Field radius (global: [-1, 1])
        rx = 2 / features.shape[-2] / 2
        ry = 2 / features.shape[-1] / 2

        features_coord = make_coord(features.shape[-2:], flatten=False).to(x.device)
        features_coord = features_coord.permute(2, 0, 1).unsqueeze(0).expand(features.shape[0], 2, *features.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = x_coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_features = F_torch.grid_sample(
                    input=features,
                    grid=coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_coord = F_torch.grid_sample(
                    input=features_coord,
                    grid=coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                rel_coord = x_coord - q_coord
                rel_coord[:, :, 0] *= features.shape[-2]
                rel_coord[:, :, 1] *= features.shape[-1]
                inputs = torch.cat([q_features, rel_coord], -1)

                # prepare cell
                rel_cell = x_cell.clone()
                rel_cell[:, :, 0] *= features.shape[-2]
                rel_cell[:, :, 1] *= features.shape[-1]
                inputs = torch.cat([inputs, rel_cell], -1)

                # basis generation
                batch_size, q = x_coord.shape[:2]
                pred = self.mlp(inputs.view(batch_size * q, -1)).view(batch_size, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t

        out = 0
        for pred, area in zip(preds, areas):
            out = out + pred * (area / tot_area).unsqueeze(-1)

        return out


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.mul(out, 0.1)
        out = torch.add(out, identity)

        return out


class _EDSR(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 64,
            channels: int = 64,
            num_blocks: int = 16,
    ) -> None:
        super(_EDSR, self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Residual blocks
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # Second layer
        self.conv2 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out = self.conv2(out)
        out = torch.add(out, out1)

        return out


class _MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels_list: list[int, int, int]):
        super(_MLP, self).__init__()
        layers = []

        last_channels = in_channels
        for hidden in hidden_channels_list:
            layers.append(nn.Linear(last_channels, hidden))
            layers.append(nn.ReLU(True))
            last_channels = hidden
        layers.append(nn.Linear(last_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape[:-1]
        out = self.layers(x.view(-1, x.shape[-1]))
        out = out.view(*shape, -1)

        return out


def liif_edsr(**kwargs: Any) -> LIIF:
    model = LIIF(encoder_arch="edsr", **kwargs)

    return model
