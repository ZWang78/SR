from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset

import data.util as Util


class TDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False,
                 steps=10):
        """
        Args:
            dataroot (str): 数据根目录。
            datatype (str): 数据类型，'lmdb' 或 'img'。
            l_resolution (int): 低分辨率图像分辨率。
            r_resolution (int): 高分辨率图像分辨率。
            split (str): 数据集划分，'train' 或 'val'。
            data_len (int): 数据长度。
            need_LR (bool): 是否需要返回低分辨率图像。
            steps (int): 退化程度数量。
        """
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.steps = steps  # 新增退化程度数量

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # 初始化数据长度
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            self.data_len = min(self.data_len, self.dataset_len) if self.data_len > 0 else self.dataset_len
        elif datatype == 'img':
            # 支持教师模型输出和退化图片路径
            self.teacher_path = Util.get_paths_from_images(f'{dataroot}/teacher_images')
            self.hr_path = Util.get_paths_from_images(f'{dataroot}/hr_{r_resolution}')
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))

            self.degraded_paths = {
                step: Util.get_paths_from_images(
                    f'{dataroot}/degraded_images/degradation_{step}'
                )
                for step in range(self.steps)
            }
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(f'{dataroot}/lr_{l_resolution}')
            self.dataset_len = len(self.hr_path)
            self.data_len = min(self.data_len, self.dataset_len) if self.data_len > 0 else self.dataset_len
        else:
            raise NotImplementedError(
                f'data_type [{datatype}] is not recognized.')

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        img_Teacher = None
        degraded_imgs = {}

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                # 读取教师模型的输出
                teacher_img_bytes = txn.get(f'teacher_{self.r_res}_{str(index).zfill(5)}'.encode('utf-8'))
                if teacher_img_bytes is None:
                    raise FileNotFoundError(f"Teacher image not found for index {index}")
                img_Teacher = Image.open(BytesIO(teacher_img_bytes)).convert("RGB")

                # 读取高分辨率图像
                hr_img_bytes = txn.get(f'hr_{self.r_res}_{str(index).zfill(5)}'.encode('utf-8'))
                if hr_img_bytes is None:
                    raise FileNotFoundError(f"HR image not found for index {index}")
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")

                # 读取退化图片
                for step in range(self.steps):
                    degraded_img_bytes = txn.get(
                        f'degraded_{step}_{self.r_res}_{str(index).zfill(5)}'.encode('utf-8'))
                    if degraded_img_bytes is not None:
                        degraded_imgs[step] = Image.open(BytesIO(degraded_img_bytes)).convert("RGB")

                if self.need_LR:
                    lr_img_bytes = txn.get(f'lr_{self.l_res}_{str(index).zfill(5)}'.encode('utf-8'))
                    if lr_img_bytes is not None:
                        img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            # 读取教师模型输出
            # img_Teacher = Image.open(self.teacher_path[index]).convert("RGB")
            # img_Teacher = img_Teacher.resize((self.r_res, self.r_res), Image.BILINEAR)

            # 读取高分辨率图像
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_HR = img_HR.resize((self.r_res, self.r_res), Image.BILINEAR)

            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            img_SR = img_SR.resize((self.r_res, self.r_res), Image.BILINEAR)

            # 读取退化图片
            for step in range(self.steps):
                degraded_path = self.degraded_paths[step][index]
                degraded_img = Image.open(degraded_path).convert("RGB")
                degraded_img = degraded_img.resize((self.r_res, self.r_res), Image.BILINEAR)
                degraded_imgs[step] = degraded_img

            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
                img_LR = img_LR.resize((self.r_res, self.r_res), Image.BILINEAR)

        # 转换图像数据格式
        # img_Teacher = Util.transform_augment([img_Teacher], split=self.split, min_max=(-1, 1))[0]


        img_HR = Util.transform_augment([img_HR], split=self.split, min_max=(-1, 1))[0]
        img_SR = Util.transform_augment([img_SR], split=self.split, min_max=(-1, 1))[0]
        if self.need_LR:
            img_LR = Util.transform_augment([img_LR], split=self.split, min_max=(-1, 1))[0]
        degraded_imgs = {
            step: Util.transform_augment([img], split=self.split, min_max=(-1, 1))[0]
            for step, img in degraded_imgs.items()
        }


        if self.need_LR:
            return {
                'LR': img_LR,  # 教师模型输出
                'HR': img_HR,  # 高分辨率图像
                'SR': img_SR,
                'Degraded': degraded_imgs,  # 不同阶段退化图像
                # 'Index': index,
                'Filename': self.hr_path[index].split('\\')[-1]
            }
        else:
            return {
                # 'Teacher': img_Teacher,  # 教师模型输出
                'HR': img_HR,  # 高分辨率图像
                'SR': img_SR,
                'Degraded': degraded_imgs,  # 不同阶段退化图像
                # 'Index': index,
                'Filename': self.hr_path[index].split('\\')[-1]
            }

        # img_LR = Util.transform_augment([img_LR], split=self.split, min_max=(-1, 1))[0]

        # 对所有图像进行一次调用
        # [img_SR, img_HR, degraded_imgs] = Util.transform_augment([img_SR, img_HR] + [img for step, img in degraded_imgs.items()], split=self.split, min_max=(-1, 1))

        # 返回结果


if __name__ == "__main__":
    dataset = TDataset(dataroot="C:/Users/wz/PycharmProjects/SR3_Zhe/dataset/7T_test_128_128", l_resolution=128,
                       datatype="img", steps=10)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        print(f"Sample {idx}: HR={sample['Filename']}")
        for step, degraded_img in sample['Degraded'].items():
            print(f"  Degradation {step}: Exists=True")
