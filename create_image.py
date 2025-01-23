import argparse
import logging
import os
import cv2
import numpy as np
import torch
from tensorboardX import SummaryWriter

import core.logger as Logger
import core.metrics as Metrics
import data as Data
import model as Model
from core.wandb_logger import WandbLogger
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

if __name__ == "__main__":
    # torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_256_256.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)

    # model
    # Teacher Model
    teacher_model = Model.create_model(opt)

    # Define degradation steps
    steps = 10
    degradation_coefficients = np.linspace(0, 1, steps)[::-1]

    # Create directories to save images
    teacher_output_dir = 'teacher_images'
    os.makedirs(teacher_output_dir, exist_ok=True)
    degraded_output_root = 'degraded_images'
    os.makedirs(degraded_output_root, exist_ok=True)

    # Transformation for image loading
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for _, train_data in enumerate(train_loader):
        # Get filenames from train_data
        filenames = train_data['filename']  # Assuming dataset returns 'filename'

        # Teacher model inference
        teacher_model.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')

        teacher_model.feed_data(train_data, mode='teacher')
        teacher_model.test(continous=False)
        visuals = teacher_model.get_current_visuals(need_LR=True)
        final_output = visuals['SR']  # Shape: (batch_size, C, H, W)

        # Save teacher model outputs
        for idx in range(final_output.shape[0]):
            img_tensor = final_output[idx]
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

            # img_np 的值范围为 [-1, 1]
            img_np = ((img_np + 1) / 2 * 255.0).clip(0, 255).astype(np.uint8)

            # Get filename
            filename = filenames[idx]

            # Save image
            img_pil = Image.fromarray(img_np)
            print(os.path.join(teacher_output_dir, filename))
            img_pil.save(os.path.join(teacher_output_dir, filename))

        # Save degraded images
        for coeff_idx, coeff in enumerate(degradation_coefficients):
            degradation_dir = os.path.join(degraded_output_root, f'degradation_{coeff_idx}')
            os.makedirs(degradation_dir, exist_ok=True)

            for idx in range(final_output.shape[0]):
                img_tensor = final_output[idx]
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

                # Apply Gaussian blur
                sigma = coeff * 2
                kernel_size = max(3, int(2 * round(3 * sigma) + 1))
                if kernel_size % 2 == 0:
                    kernel_size += 1

                if sigma == 0:
                    degraded_img = img_np.copy()
                else:
                    degraded_img = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)

                # Denormalize
                # 假设 degraded_img 的值范围为 [-1, 1]
                degraded_img = ((degraded_img + 1) / 2 * 255.0).clip(0, 255).astype(np.uint8)

                # Get filename
                filename = filenames[idx]

                # Save degraded image
                img_pil = Image.fromarray(degraded_img)
                img_pil.save(os.path.join(degradation_dir, filename))