"""DnCNN / Restormer 共用的固定 Noisy/GT 数据读取；不裁剪、增强、加噪。

只接受 train / val，禁止把测试集接入训练或选权重。
DIV2K 严格读取现有 noise_manifest.csv，训练三档 sigma 各1750对，
验证每档400对；SIDD读取既有训练块与官方验证块。
"""

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DenoisingDataset(Dataset):
    def __init__(self, data_root, dataset, split):
        if dataset not in ("div2k", "sidd") or split not in ("train", "val"):
            raise ValueError("dataset 必须为 div2k/sidd，split 必须为 train/val")
        self.root = Path(data_root).resolve()
        self.dataset, self.split = dataset, split
        self.patch_size = 128 if split == "train" else 256
        self.samples = []  # (噪声路径，GT 路径，分组标签)
        self.source_ids = set()
        digest = hashlib.sha256()
        if dataset == "div2k":
            self._div2k(digest)
        else:
            self._sidd()
        for noisy, gt, group in self.samples:
            for path in (noisy, gt):
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"图片缺失或为空：{path}")
                # 不在启动时读取全部图片；用清单、配对路径和文件大小记录数据版本。
                digest.update(f"{path.relative_to(self.root).as_posix()}|{path.stat().st_size}\n".encode())
            digest.update(f"group={group}\n".encode())
        self.fingerprint = digest.hexdigest()

    def _div2k(self, digest):
        prefix = f"div2k_{self.split}"
        manifest = self.root / prefix / "noise_manifest.csv"
        digest.update(manifest.read_bytes())
        assignments, patches = defaultdict(list), defaultdict(set)
        seen = set()
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                sample, source, sigma = row["sample_id"], row["source_image_id"], int(row["sigma"])
                patch_count = 7 if self.split == "train" else 4
                if (row["split"] != self.split or sigma not in (15, 25, 50)
                        or re.fullmatch(r"\d{4}", source) is None
                        or sample not in {f"{source}_p{i:02d}" for i in range(1, patch_count + 1)}):
                    raise ValueError(f"DIV2K 清单样本不合法：{row}")
                noise_dir = "noise" if self.split == "train" else f"noise/sigma_{sigma}"
                expected_noise = f"{prefix}/{noise_dir}/{sample}.png"
                expected_gt = f"{prefix}/gt/{sample}.png"
                if (row["noise_path"].replace("\\", "/") != expected_noise
                        or row["gt_path"].replace("\\", "/") != expected_gt
                        or expected_noise in seen):
                    raise ValueError(f"DIV2K 路径/配对/重复记录异常：{sample}")
                seen.add(expected_noise)
                self.samples.append((self.root / expected_noise, self.root / expected_gt, str(sigma)))
                assignments[sample].append(sigma)
                patches[source].add(sample)
                self.source_ids.add(source)
        per_sigma, sources, per_source = (1750, 750, 7) if self.split == "train" else (400, 100, 4)
        counts = Counter(group for _, _, group in self.samples)
        if counts != Counter({str(sigma): per_sigma for sigma in (15, 25, 50)}):
            raise ValueError(f"DIV2K {self.split} 三档数量不符合现有方案：{counts}")
        if len(patches) != sources or any(len(value) != per_source for value in patches.values()):
            raise ValueError("DIV2K 原图数量或每图 Patch 数量不符合现有方案")
        if self.split == "train":
            if any(len(value) != 1 for value in assignments.values()):
                raise ValueError("每张 DIV2K 训练 GT 必须只对应一张 noise")
        elif any(sorted(value) != [15, 25, 50] for value in assignments.values()):
            raise ValueError("每张 DIV2K 验证 GT 必须对应三档 noise")
        expected_gt_names = {f"{sample}.png" for sample in assignments}
        if {path.name for path in (self.root / prefix / "gt").glob("*.png")} != expected_gt_names:
            raise ValueError("DIV2K GT 目录与清单不一致")
        for sigma in (15, 25, 50):
            directory = self.root / prefix / ("noise" if self.split == "train" else f"noise/sigma_{sigma}")
            expected = {path.name for path, _, group in self.samples if self.split == "train" or group == str(sigma)}
            if {path.name for path in directory.glob("*.png")} != expected:
                raise ValueError(f"DIV2K noise 目录与清单不一致：{directory}")
            if self.split == "train":
                break

    def _sidd(self):
        directory = self.root / f"sidd_{self.split}"
        noisy_dir, gt_dir = ((directory / "noisy", directory / "gt") if self.split == "train"
                            else (directory / "input_crops", directory / "target_crops"))
        noisy = {path.name: path for path in noisy_dir.glob("*.png")}
        gt = {path.name: path for path in gt_dir.glob("*.png")}
        expected_count = 5184 if self.split == "train" else 1280
        if set(noisy) != set(gt) or len(noisy) != expected_count:
            raise ValueError(f"SIDD {self.split} 要求同名配对 {expected_count} 对，"
                             f"当前 noisy={len(noisy)}，GT={len(gt)}")
        for name in sorted(noisy):
            if self.split == "val" and re.fullmatch(r"\d{4}-\d{4}\.png", name) is None:
                raise ValueError(f"SIDD 官方验证块名称不合法：{name}")
            if self.split == "train":
                if re.fullmatch(r".+_(010|011)_p(0[1-9]|1[0-8])\.png", name) is None:
                    raise ValueError(f"SIDD 训练块名称不合法：{name}")
                self.source_ids.add(name.rsplit("_p", 1)[0])
            self.samples.append((noisy[name], gt[name], "real"))
        # 官方 val 块的名称没有原始场景映射，不能用它声称原图/场景独立。

    def __len__(self):
        return len(self.samples)

    def _read(self, path):
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (self.patch_size, self.patch_size):
                raise ValueError(f"要求 RGB {self.patch_size}×{self.patch_size}：{path}")
            array = np.array(image, dtype=np.uint8, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)

    def __getitem__(self, index):
        noisy, gt, group = self.samples[index]
        return self._read(noisy), self._read(gt), group
