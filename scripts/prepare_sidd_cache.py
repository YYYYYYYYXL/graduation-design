"""SIDD 缓存维护：默认仅检查；明确指定 --build-missing / --rebuild 才写入。"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "splits" / "sidd_split.csv"
CACHE_ROOT = ROOT / "data" / "cache"


def read_rows():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"缺少数据划分 CSV：{CSV_PATH}")
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "val")
    }


def resolve(relative_path):
    path = Path(relative_path.replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def paths(split, index):
    folder = CACHE_ROOT / ("sidd_full" if split == "train" else "sidd_val")
    # 与已有训练脚本兼容：历史训练缓存沿用 test_ 前缀，它不属于测试集。
    prefix = "test" if split == "train" else "val"
    return (folder / f"{prefix}_{index:04d}_noisy.npy",
            folder / f"{prefix}_{index:04d}_gt.npy")


def inspect_cache(path, split):
    if not path.is_file():
        return "缺失"
    try:
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if data.dtype != np.uint8 or data.ndim != 3 or data.shape[2] != 3:
            return f"类型或形状异常：{data.dtype} {data.shape}"
        if split == "val" and data.shape != (256, 256, 3):
            return f"验证裁剪尺寸异常：{data.shape}"
        return None
    except (OSError, ValueError, EOFError) as exc:
        return str(exc)


def write_npy_safely(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".building")
    try:
        with temp.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def create_pair(row, split):
    with Image.open(resolve(row["noisy_path"])) as image:
        if image.mode != "RGB":
            raise ValueError(f"NOISY 不是 RGB：{row['noisy_path']}")
        noisy = np.asarray(image, dtype=np.uint8)
    with Image.open(resolve(row["gt_path"])) as image:
        if image.mode != "RGB":
            raise ValueError(f"GT 不是 RGB：{row['gt_path']}")
        gt = np.asarray(image, dtype=np.uint8)
    if noisy.shape != gt.shape:
        raise ValueError(f"NOISY / GT 尺寸不一致：{row['noisy_path']}")
    if split == "val":
        height, width, _ = noisy.shape
        if min(height, width) < 256:
            raise ValueError("原图小于 256×256")
        top, left = (height - 256) // 2, (width - 256) // 2
        noisy = noisy[top:top + 256, left:left + 256]
        gt = gt[top:top + 256, left:left + 256]
    return noisy, gt


def main():
    parser = argparse.ArgumentParser(description="SIDD 训练与验证缓存检查 / 构建")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-missing", action="store_true", help="只补建缺失缓存")
    mode.add_argument("--rebuild", action="store_true", help="从原始 PNG 重建全部缓存，会覆盖旧 .npy")
    args = parser.parse_args()
    rows = read_rows()
    csv_sha = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    print("当前 CSV SHA256：", csv_sha)
    failures = []
    for split, samples in rows.items():
        print(f"\n检查 {split}：{len(samples)} 对")
        manifest_path = CACHE_ROOT / ("sidd_full" if split == "train" else "sidd_val") / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("csv_sha256") != csv_sha:
                raise RuntimeError(f"{split} 缓存的 CSV 校验和与现有 CSV 不一致；停止操作")
        else:
            print("注意：历史缓存没有 manifest；只能检查数量和形状，不能证明与 CSV 一一对应")
        for idx, row in enumerate(samples):
            noisy_path, gt_path = paths(split, idx)
            e1, e2 = inspect_cache(noisy_path, split), inspect_cache(gt_path, split)
            should_build = args.rebuild or (args.build_missing and (e1 == "缺失" or e2 == "缺失"))
            if should_build:
                if not args.rebuild and any(e not in (None, "缺失") for e in (e1, e2)):
                    failures.append(f"{split} 第 {idx} 对存在损坏缓存；须先排查：{e1} {e2}")
                    continue
                print(f"构建 {split} {idx + 1}/{len(samples)}", flush=True)
                noisy, gt = create_pair(row, split)
                write_npy_safely(noisy_path, noisy)
                write_npy_safely(gt_path, gt)
                e1, e2 = inspect_cache(noisy_path, split), inspect_cache(gt_path, split)
            if e1 is not None or e2 is not None:
                failures.append(f"{split} 第 {idx} 对：NOISY={e1}, GT={e2}")
        # 补缺操作不能证明所有旧缓存和 CSV 对应，不给旧缓存补签名。
        if args.rebuild and not any(x.startswith(split + " ") for x in failures):
            manifest_path.write_text(json.dumps({
                "csv_sha256": csv_sha, "split": split, "count": len(samples),
                "prefix": "test" if split == "train" else "val",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        print("\n发现缓存问题（最多显示前 10 项）：")
        for message in failures[:10]:
            print(" -", message)
        raise SystemExit(f"未通过检查：{len(failures)} 项")
    print("\n缓存检查通过（文件存在，dtype 与 shape 合法；旧缓存索引匹配仍须注意）")


if __name__ == "__main__":
    main()
