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
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
from ptflops import get_model_complexity_info  # 用于计算FLOPs和参数数量
from fvcore.nn import FlopCountAnalysis, parameter_count

def compute_flops_with_fvcore(model, input_tensor):
    model.eval()  # Ensure the model is in evaluation mode

    # Compute FLOPs
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()

    # Compute Parameters
    params = parameter_count(model)

    return total_flops, params

if __name__ == "__main__":
    torch.set_num_threads(1)
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

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset1(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset1(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # Initialize Student Model
    student_model_opt = opt
    student_model_opt['model']['unet']['inner_channel'] = 64  # Example: smaller student model
    student_model_opt['model']['unet']['channel_multiplier'] = [1, 2, 2, 4, 4, 8, 8, 16]
    student_model_opt['model']['unet']['attn_res'] = [4, 8, 16, 32]
    student_model_opt['path']['resume_state'] = None
    student_model_opt['model']['beta_schedule']['train']['n_timestep'] = 2000
    student_model_opt['model']['beta_schedule']['val']['n_timestep'] = 2000
    student_model = Model.create_model(student_model_opt)
    logger.info('Models Initialized')

    # -----------------------Teacher---------------------



    steps = 10

    # Generate N coefficients from 0 to 1
    degradation_coefficients = np.linspace(0, 1, steps)[::-1]

    current_step = student_model.begin_step
    current_epoch = student_model.begin_epoch
    n_iter = opt['train']['n_iter']

    while current_step < n_iter:

        current_epoch += 1
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            N = min(current_step // (n_iter // steps), steps-1)
            # print(N)
            N = 5
            train_data_student = train_data
            train_data_student['HR'] = train_data['Degraded'][N]  # shape: (batch_size, C, H, W)


            if opt['path']['resume_state']:
                logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                    current_epoch, current_step))

            student_model.set_new_noise_schedule(
                opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

            ########################
            # import matplotlib.pyplot as plt
            # import torch
            #
            # # 假设 train_data_student 已加载
            # # 查看 train_data_student 的内容
            # print("Keys in train_data_student:", train_data_student.keys())
            #
            # # 检查 HR 图像
            # if True:
            #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            #     hr_image = train_data_student['HR'][5]  # 取第一个样本
            #     print("HR Image Shape:", hr_image.shape)  # 检查形状
            #
            #     # 转换 Tensor [C, H, W] -> NumPy [H, W, C]
            #     hr_image = hr_image.permute(1, 2, 0).cpu().numpy()
            #     hr_image = ((hr_image + 1) / 2 * 255).astype('uint8')
            #
            #     # 显示图像
            #     axs[0].imshow(hr_image.astype('uint8'))  # 转为 uint8 格式
            #     axs[0].set_title("HR Image")
            #     axs[0].axis("off")
            #
            # # 显示更多内容（如 LR 或 Degraded 图像）
            #
            #
            #
            #     # HR 图像
            #     hr_image = train_data_student['SR'][5].permute(1, 2, 0).cpu().numpy()
            #     hr_image = ((hr_image + 1) / 2 * 255).astype('uint8')
            #     axs[1].imshow(hr_image.astype('uint8'))
            #     axs[1].set_title("HR Image")
            #     axs[1].axis("off")
            #
            #     # Degraded 图像
            #     degraded_image = train_data_student['Degraded'][5][5].permute(1, 2, 0).cpu().numpy()  # 取第一个降质图像
            #     degraded_image = ((degraded_image + 1) / 2 * 255).astype('uint8')
            #     axs[2].imshow(degraded_image.astype('uint8'))
            #     axs[2].set_title("Degraded Image")
            #     axs[2].axis("off")
            #
            #     plt.tight_layout()
            #     plt.show()

            ####################### FLOPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
            # input_tensor={}
            # input_tensor['HR'] = torch.randn(1, 3, 128, 128).cuda() # Example input tensor
            # input_tensor['SR'] = torch.randn(1, 3, 128, 128).cuda()  # Example input tensor
            # flops, params = compute_flops_with_fvcore(student_model.netG, input_tensor)
            # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
            # print(f"Parameters: {params[''] / 1e6:.2f} M")


            if opt['phase'] == 'train':
                student_model.feed_data(train_data_student)
                student_model.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                # if True:
                    print(N)
                    logs = student_model.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    student_model.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        val_data_student = val_data
                        val_data_student['HR'] = val_data['Degraded'][N]


                        ###################################################
                        # import matplotlib.pyplot as plt
                        # import torch
                        #
                        # # 假设 train_data_student 已加载
                        # # 查看 train_data_student 的内容
                        # print("Keys in val_data_student:", val_data_student.keys())
                        #
                        # # 检查 HR 图像
                        # if True:
                        #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                        #     hr_image = val_data_student['HR'][0]  # 取第一个样本
                        #     print("HR Image Shape:", hr_image.shape)  # 检查形状
                        #
                        #     # 转换 Tensor [C, H, W] -> NumPy [H, W, C]
                        #     hr_image = hr_image.permute(1, 2, 0).cpu().numpy()
                        #     hr_image = ((hr_image + 1) / 2 * 255).astype('uint8')
                        #
                        #     # 显示图像
                        #     axs[0].imshow(hr_image.astype('uint8'))  # 转为 uint8 格式
                        #     axs[0].set_title("HR Image")
                        #     axs[0].axis("off")
                        #
                        # # 显示更多内容（如 LR 或 Degraded 图像）
                        #
                        #
                        #
                        #     # HR 图像
                        #     hr_image = val_data_student['SR'][0].permute(1, 2, 0).cpu().numpy()
                        #     hr_image = ((hr_image + 1) / 2 * 255).astype('uint8')
                        #     axs[1].imshow(hr_image.astype('uint8'))
                        #     axs[1].set_title("HR Image")
                        #     axs[1].axis("off")
                        #
                        #     # Degraded 图像
                        #     degraded_image = val_data_student['Degraded'][5][0].permute(1, 2, 0).cpu().numpy()  # 取第一个降质图像
                        #     degraded_image = ((degraded_image + 1) / 2 * 255).astype('uint8')
                        #     axs[2].imshow(degraded_image.astype('uint8'))
                        #     axs[2].set_title("Degraded Image")
                        #     axs[2].axis("off")
                        #
                        #     plt.tight_layout()
                        #     plt.show()


                        ###################################################

                        student_model.feed_data(val_data_student)
                        student_model.test(continous=False)
                        visuals = student_model.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

                        #fake_img = np.expand_dims(fake_img, axis=-1)
                        #sr_img = np.expand_dims(sr_img, axis=-1)
                        #hr_img = np.expand_dims(hr_img, axis=-1)

                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                            idx)
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    student_model.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    student_model.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)