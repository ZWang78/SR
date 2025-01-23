import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2


def degrade_image_multiple_steps(image_path, save_dir, num_steps=10):
    """
    从退化一半的质量开始，逐步退化图片并保存。

    Args:
        image_path (str): 原始图片的路径。
        save_dir (str): 保存退化图片的目录。
        num_steps (int): 退化的步数，默认是 10。

    Returns:
        List[torch.Tensor]: 每一步退化的图像张量列表。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 转换工具
    transform_to_tensor = transforms.ToTensor()
    transform_to_image = transforms.ToPILImage()

    # 加载原始图片
    original_image = Image.open(image_path).convert("RGB")
    original_tensor = transform_to_tensor(original_image)

    # 初始化退化程度
    degradation_coefficients = np.linspace(1, 0.0, num_steps)  # 从 0.5 到 0，均匀退化
    degraded_tensors = []

    for step, degradation_coefficient in enumerate(degradation_coefficients):
        # 添加高斯模糊噪声
        sigma = degradation_coefficient * 2  # 标准差与退化系数相关
        kernel_size = max(3, int(2 * round(3 * sigma) + 1))  # 确定核大小
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保核大小为奇数

        # 应用高斯模糊
        if sigma > 0:
            degraded_np = cv2.GaussianBlur(
                original_tensor.permute(1, 2, 0).numpy(),  # 转换为 HWC 格式
                (kernel_size, kernel_size),
                sigma
            )
        else:
            degraded_np = original_tensor.permute(1, 2, 0).numpy()

        # 归一化到 [0, 255]，并转换为 uint8 类型
        degraded_np = (degraded_np * 255).clip(0, 255).astype(np.uint8)

        # 保存退化图像
        degraded_image = Image.fromarray(degraded_np)
        save_path = os.path.join(save_dir, f"degraded_step_{step:02d}.png")
        degraded_image.save(save_path)

        # 转换回张量并存储
        degraded_tensors.append(transform_to_tensor(degraded_image))

    return degraded_tensors


# 示例用法
if __name__ == "__main__":
    # 原始图片路径
    image_path = r"C:\Users\wz\PycharmProjects\SR3_Zhe\dataset\clinical_paper3\3_7T.png"
    # 保存退化图片的目录
    save_dir = r"C:\Users\wz\PycharmProjects\SR3_Zhe\dataset\clinical_paper3/3_7T"
    # 退化步数
    num_steps = 20

    # 执行退化处理
    degraded_tensors = degrade_image_multiple_steps(image_path, save_dir, num_steps)
