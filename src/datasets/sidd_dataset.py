
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SIDDDataset(Dataset):

    def __init__(
        self,
        split="train",
        patch_size=128,
        eval_patch_size=256,
        patches_per_image=1,
        project_root=None
    ):
        # 1. 检查数据集类型
        if split not in ("train", "val", "test"):
            raise ValueError(f"不支持的数据集类型：{split}")

        if patches_per_image < 1:
            raise ValueError("patches_per_image 必须大于等于 1")

        self.split = split
        self.patch_size = patch_size
        self.eval_patch_size = eval_patch_size
        self.patches_per_image = patches_per_image

        # 2. 定位项目根目录
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root).resolve()

        # 验证集预处理缓存目录
        self.val_cache_dir = (
            self.project_root
            / "data"
            / "cache"
            / "sidd_val"
        )

        # 3. 定位 CSV 文件
        csv_path = (
            self.project_root
            / "data"
            / "splits"
            / "sidd_split.csv"
        )

        if not csv_path.is_file():
            raise FileNotFoundError(f"找不到划分文件：{csv_path}")

        # 4. 读取对应 split 的图片路径
        self.samples = []

        with csv_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:
                if row["split"] != split:
                    continue

                noisy_path = self._resolve_path(
                    row["noisy_path"]
                )

                gt_path = self._resolve_path(
                    row["gt_path"]
                )

                if not noisy_path.is_file():
                    raise FileNotFoundError(noisy_path)

                if not gt_path.is_file():
                    raise FileNotFoundError(gt_path)

                self.samples.append((noisy_path, gt_path))

        if not self.samples:
            raise ValueError(f"{split} 没有找到样本")

    def _resolve_path(self, path_string):
        # 兼容 CSV 中 Windows 风格的相对路径
        path = Path(path_string.replace("\\", "/"))

        if not path.is_absolute():
            path = self.project_root / path

        return path

    def __len__(self):
        # 训练时允许每张原图随机采样多个 Patch
        if self.split == "train":
            return len(self.samples) * self.patches_per_image

        return len(self.samples)

    def __getitem__(self, index):

        # 1. 找到原始图片
        if self.split == "train":
            index = index // self.patches_per_image

        noisy_path, gt_path = self.samples[index]

            # 验证集优先读取预处理缓存
        if self.split == "val" and self.val_cache_dir.is_dir():

            noisy_cache = self.val_cache_dir / f"val_{index:04d}_noisy.npy"
            gt_cache = self.val_cache_dir / f"val_{index:04d}_gt.npy"

            if not noisy_cache.is_file() or not gt_cache.is_file():
                raise FileNotFoundError(
                    f"验证集缓存不完整，索引：{index}"
                )

            noisy = np.load(noisy_cache, allow_pickle=False)
            gt = np.load(gt_cache, allow_pickle=False)

            expected_shape = (
                self.eval_patch_size,
                self.eval_patch_size,
                3
            )

            if noisy.shape != expected_shape or gt.shape != expected_shape:
                raise ValueError(
                    f"验证集缓存尺寸异常，索引：{index}"
                )

            return self._to_tensor(noisy), self._to_tensor(gt)

        # 2. 读取图片
        with Image.open(noisy_path) as img:
            noisy = img.copy()

        with Image.open(gt_path) as img:
            gt = img.copy()

        # 3. 检查图片
        if noisy.size != gt.size:
            raise ValueError(
                f"NOISY/GT 尺寸不一致：{noisy_path}"
            )

        if noisy.mode != "RGB" or gt.mode != "RGB":
            raise ValueError(
                f"图片不是 RGB：{noisy_path}"
            )

        # 4. 裁剪
        if self.split == "train":
            noisy, gt = self._crop_pair(
                noisy, gt, self.patch_size, random_crop=True
            )

            # 仅训练集进行随机增强
            noisy, gt = self._augment_pair(noisy, gt)

        else:
            # 验证和测试使用固定中心裁剪
            noisy, gt = self._crop_pair(
                noisy, gt, self.eval_patch_size, random_crop=False
            )

        # 5. 转换 Tensor
        noisy = self._to_tensor(noisy)
        gt = self._to_tensor(gt)

        return noisy, gt

    @staticmethod
    def _crop_pair(noisy, gt, size, random_crop=True):

        width, height = noisy.size

        if width < size or height < size:
            raise ValueError("图片尺寸小于 Patch 尺寸")

        if random_crop:
            left = random.randint(0, width - size)
            top = random.randint(0, height - size)
        else:
            left = (width - size) // 2
            top = (height - size) // 2

        box = (
            left,
            top,
            left + size,
            top + size
        )

        return noisy.crop(box), gt.crop(box)

    @staticmethod
    def _augment_pair(noisy, gt):

        # 随机水平翻转
        if random.random() < 0.5:
            noisy = noisy.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # 随机垂直翻转
        if random.random() < 0.5:
            noisy = noisy.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            gt = gt.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # 随机旋转 0、90、180、270 度
        angle = random.choice([0, 90, 180, 270])

        if angle != 0:
            noisy = noisy.rotate(angle)
            gt = gt.rotate(angle)

        return noisy, gt

    @staticmethod
    def _to_tensor(image):

        array = np.asarray(image, dtype=np.float32) / 255.0

        # HWC -> CHW
        array = array.transpose(2, 0, 1).copy()

        return torch.from_numpy(array)
