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
import os

import cv2
import numpy as np
import torch
from natsort import natsorted

import config
import imgproc
import model
from image_quality_assessment import PSNR, SSIM
from utils import make_coord
from utils import make_directory

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main() -> None:
    # Initialize the super-resolution sr_model
    sr_model = model.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                      encoder_channels=config.encoder_channels,
                                                      out_channels=config.out_channels)
    sr_model = sr_model.to(device=config.device)
    print(f"Build `{config.model_arch_name}` model successfully.")

    # Load the super-resolution sr_model weights
    checkpoint = torch.load(config.model_weights_path, map_location=lambda storage, loc: storage)
    sr_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{config.model_arch_name}` model weights `{os.path.abspath(config.model_weights_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    make_directory(config.sr_dir)

    # Start the verification mode of the sr_model.
    sr_model.eval()

    # Initialize the sharpness evaluation function
    psnr_model = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim_model = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Set the sharpness evaluation function calculation device to the specified sr_model
    psnr_model = psnr_model.to(device=config.device, non_blocking=True)
    ssim_model = ssim_model.to(device=config.device, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.gt_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        gt_image_path = os.path.join(config.gt_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        lr_image_path = os.path.join(config.lr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(gt_image_path)}`...")
        # Read LR image and HR image
        gt_image = cv2.imread(gt_image_path).astype(np.float32) / 255.0
        lr_image = cv2.imread(lr_image_path).astype(np.float32) / 255.0

        # Convert BGR channel image format data to RGB channel image format data
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert RGB channel image format data to Tensor channel image format data
        gt_tensor = imgproc.image_to_tensor(gt_image, False, False)
        lr_tensor = imgproc.image_to_tensor(lr_image, False, False)
        gt_tensor_coord = make_coord(gt_tensor.contiguous().shape[-2:])
        gt_tensor_contiguous = gt_tensor.contiguous().view(3, -1).permute(1, 0)
        gt_tensor_cell = torch.ones_like(gt_tensor_coord)
        gt_tensor_cell[:, 0] *= 2 / gt_tensor.shape[-2]
        gt_tensor_cell[:, 1] *= 2 / gt_tensor.shape[-1]

        # Transfer Tensor channel image format data to CUDA device
        gt_tensor_contiguous = gt_tensor_contiguous.unsqueeze_(0).to(device=config.device, non_blocking=True)
        gt_tensor_coord = gt_tensor_coord.unsqueeze_(0).to(device=config.device, non_blocking=True)
        gt_tensor_cell = gt_tensor_cell.unsqueeze_(0).to(device=config.device, non_blocking=True)
        lr_tensor = lr_tensor.unsqueeze_(0).to(device=config.device, non_blocking=True)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = sr_model(lr_tensor, gt_tensor_coord, gt_tensor_cell)

        # N,C,HW to N,C,H,W
        batch_size, channels, lr_image_height, lr_image_width = lr_tensor.shape
        shape = [batch_size,
                 round(lr_image_height * config.upscale_factor),
                 round(lr_image_width * config.upscale_factor),
                 channels]
        sr_tensor = sr_tensor.view(*shape).permute(0, 3, 1, 2).contiguous()
        gt_tensor = gt_tensor_contiguous.view(*shape).permute(0, 3, 1, 2).contiguous()

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)

        # Cal IQA metrics
        psnr_metrics += psnr_model(sr_tensor, gt_tensor).item()
        ssim_metrics += ssim_model(sr_tensor, gt_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files

    print(f"PSNR: {avg_psnr:4.2f} [dB]\n"
          f"SSIM: {avg_ssim:4.4f} [u]")


if __name__ == "__main__":
    main()
