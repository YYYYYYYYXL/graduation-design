"""按现有 noise_manifest.csv 读取 DIV2K；不再裁剪或重新添加噪声。"""

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SIGMAS = (15, 25, 50)


class DIV2KPretrainDataset(Dataset):
    """train: 5250 对 128×128；val: 1200 对 256×256（三个 sigma）。"""

    def __init__(self, data_root, split):
        if split not in ("train", "val"):
            raise ValueError("预训练只读取 train/val，不读取测试集")
        self.data_root = Path(data_root).resolve()
        self.split = split
        self.patch_size = 128 if split == "train" else 256
        manifest = self.data_root / f"div2k_{split}" / "noise_manifest.csv"
        self.manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.samples = []
        self.source_ids = set()
        assignments = defaultdict(list)
        source_patches = defaultdict(set)
        seen_noise = set()
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["split"] != split:
                    raise ValueError(f"{manifest} 中混入其他 split")
                sample_id, source_id = row["sample_id"], row["source_image_id"]
                if not re.fullmatch(r"\d{4}", source_id):
                    raise ValueError(f"原图编号不合法：{source_id}")
                patch_count = 7 if split == "train" else 4
                expected_ids = {f"{source_id}_p{i:02d}" for i in range(1, patch_count + 1)}
                if sample_id not in expected_ids:
                    raise ValueError(f"Patch 名称不合法：{sample_id}")
                sigma = int(row["sigma"])
                if sigma not in SIGMAS:
                    raise ValueError(f"不支持的 sigma：{sigma}")
                prefix = f"div2k_{split}"
                expected_gt = f"{prefix}/gt/{sample_id}.png"
                noise_dir = "noise" if split == "train" else f"noise/sigma_{sigma}"
                expected_noise = f"{prefix}/{noise_dir}/{sample_id}.png"
                if (row["gt_path"].replace("\\", "/") != expected_gt
                        or row["noise_path"].replace("\\", "/") != expected_noise):
                    raise ValueError(f"清单路径或 Noise/GT 配对不符：{sample_id}")
                if expected_noise in seen_noise:
                    raise ValueError(f"重复噪声样本：{expected_noise}")
                seen_noise.add(expected_noise)
                noisy_path = self.data_root / expected_noise
                gt_path = self.data_root / expected_gt
                for path in (noisy_path, gt_path):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                self.samples.append((noisy_path, gt_path, sigma))
                assignments[sample_id].append(sigma)
                source_patches[source_id].add(sample_id)
                self.source_ids.add(source_id)
        counts = Counter(sigma for _, _, sigma in self.samples)
        expected_per_sigma = 1750 if split == "train" else 400
        if counts != Counter({sigma: expected_per_sigma for sigma in SIGMAS}):
            raise ValueError(f"{split} 数量不符合现有方案：{dict(counts)}")
        if split == "train" and any(len(values) != 1 for values in assignments.values()):
            raise ValueError("每张训练 GT 必须恰好对应一张 Noise")
        if split == "val" and any(sorted(values) != list(SIGMAS) for values in assignments.values()):
            raise ValueError("每张验证 GT 必须分别对应 sigma=15/25/50")
        expected_sources = 750 if split == "train" else 100
        if len(source_patches) != expected_sources or any(
            len(patches) != (7 if split == "train" else 4)
            for patches in source_patches.values()
        ):
            raise ValueError(f"{split} 原图数量或每张原图的 Patch 数量不符")

    def __len__(self):
        return len(self.samples)

    def _read_tensor(self, path):
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (self.patch_size, self.patch_size):
                raise ValueError(f"要求 RGB {self.patch_size}×{self.patch_size}：{path}")
            array = np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)

    def __getitem__(self, index):
        noisy_path, gt_path, sigma = self.samples[index]
        return self._read_tensor(noisy_path), self._read_tensor(gt_path), sigma
