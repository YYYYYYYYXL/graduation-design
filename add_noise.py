"""给已裁剪的 DIV2K GT 离线添加高斯噪声，不修改 GT 或原图划分。

在 denoise310 环境、项目根目录运行：
    python add_noise.py
    python add_noise.py --split train

训练：5250张GT各配一张Noisy；sigma=15/25/50严格各1750张。
验证：400张GT各配三档Noisy，共1200张。
测试：200张GT各配三档Noisy，共600张。
验证和测试的输出位于 noise/sigma_15、sigma_25、sigma_50。
固定种子并保存分配清单；同配置重跑跳过已有非空输出。
依赖：NumPy、Pillow。
"""

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


DATA = Path(__file__).resolve().parent / "data"
SIGMAS = (15, 25, 50)
SEED = 20261015
SETTINGS = {
    "train": {"originals": 750, "patches": 7, "size": 128},
    "val": {"originals": 100, "patches": 4, "size": 256},
    "test": {"originals": 50, "patches": 4, "size": 256},
}
MANIFEST_FIELDS = [
    "split", "sample_id", "source_image_id", "gt_path", "noise_path",
    "sigma", "noise_seed",
]


def relative(path):
    """所有清单路径以 data/ 为基准，统一使用正斜杠。"""
    return path.relative_to(DATA).as_posix()


def read_original_splits():
    groups = {split: set() for split in SETTINGS}
    with (DATA / "div2k_split.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            if split not in groups:
                raise ValueError(f"CSV 中有未知划分：{split}")
            image_id = Path(row["image"].replace("\\", "/")).stem
            if image_id in groups[split]:
                raise ValueError(f"CSV 中有重复原图：{split}/{image_id}")
            groups[split].add(image_id)
    for split, options in SETTINGS.items():
        if len(groups[split]) != options["originals"]:
            raise ValueError(f"CSV 的 {split} 应有 {options['originals']} 张原图，实际 {len(groups[split])} 张。")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[a] & groups[b]:
            raise ValueError(f"CSV 的 {a}/{b} 原图编号交叉。")
    return groups


def noise_seed(base_seed, split, sample_id, sigma):
    # 每个样本和sigma有独立的种子，分开运行或中断重跑均不改变结果。
    key = f"{base_seed}|{split}|{sample_id}|{sigma}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def build_job(split, original_ids, seed):
    options = SETTINGS[split]
    base = DATA / f"div2k_{split}"
    gt_dir = base / "gt"
    noise_dir = base / "noise"
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"缺少GT目录：{gt_dir}；请先运行裁剪脚本。")
    expected_names = {
        f"{image_id}_p{patch:02d}.png"
        for image_id in original_ids
        for patch in range(1, options["patches"] + 1)
    }
    images = sorted(p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    actual_names = {p.name for p in images}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"{split} GT应有{len(expected_names)}张Patch，实际{len(images)}张。"
            f"缺少示例：{missing[:3]}；额外示例：{extra[:3]}。"
            "测试集应先从50张整图裁为200张256×256 Patch。"
        )
    if any(p.stat().st_size == 0 for p in images):
        raise ValueError(f"{gt_dir} 中存在空文件。")

    if split == "train":
        # 先构造严格等量的sigma列表，再打乱；不使用独立三选一导致比例波动。
        if len(images) % len(SIGMAS):
            raise ValueError("训练GT数量不能被3整除，无法严格均分三档sigma。")
        assigned_sigmas = list(SIGMAS) * (len(images) // len(SIGMAS))
        random.Random(seed).shuffle(assigned_sigmas)
    else:
        assigned_sigmas = None

    rows = []
    for index, gt in enumerate(images):
        sigmas = (assigned_sigmas[index],) if assigned_sigmas is not None else SIGMAS
        for sigma in sigmas:
            target = (noise_dir / gt.name if split == "train"
                      else noise_dir / f"sigma_{sigma}" / gt.name)
            rows.append({
                "split": split, "sample_id": gt.stem,
                "source_image_id": gt.stem.split("_p")[0],
                "gt_path": relative(gt), "noise_path": relative(target),
                "sigma": sigma, "noise_seed": noise_seed(seed, split, gt.stem, sigma),
            })
    config = {
        "version": 1, "split": split, "base_seed": seed,
        "sigmas_8bit": list(SIGMAS), "gt_count": len(images),
        "noise_count": len(rows), "patch_size": options["size"],
        "sigma_counts": {str(sigma): sum(row["sigma"] == sigma for row in rows) for sigma in SIGMAS},
        "assignment": "equal_counts_shuffled_once" if split == "train" else "all_sigmas_per_gt",
        "noise_seed_rule": "first_8_bytes_big_endian_SHA256(seed|split|sample_id|sigma)",
        "generation": "numpy.default_rng(noise_seed).normal(0, sigma, RGB_shape)",
        "pixels": "uint8(round_to_nearest_even(clip(float64(GT) + noise, 0, 255)))",
        "numpy_version": np.__version__,
        "gt_paths": [relative(p) for p in images],
    }
    config_path = base / "noise_config.json"
    if config_path.exists():
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        if old_config != config:
            raise ValueError(f"{config_path} 与当前配置不同，停止以避免混用不同噪声数据。")
    elif noise_dir.exists() and any(p.is_file() for p in noise_dir.rglob("*")):
        raise ValueError(f"{noise_dir} 已有文件但缺少noise_config.json，不能确定其来源。")
    expected_outputs = {DATA / row["noise_path"] for row in rows}
    if noise_dir.exists():
        extra = [p for p in noise_dir.rglob("*")
                 if p.is_file() and p.suffix.lower() == ".png" and p not in expected_outputs]
        if extra:
            raise ValueError(f"{noise_dir} 有不属于当前方案的图片，例如：{extra[:3]}")
    return {"base": base, "images": images, "rows": rows, "config": config,
            "config_path": config_path, "manifest_path": base / "noise_manifest.csv"}


def write_text_safely(path, content):
    if path.exists() and path.read_bytes() == content:
        return
    temp = path.with_name(path.name + ".part")
    temp.write_bytes(content)
    temp.replace(path)


def add_gaussian_noise(clean, sigma, seed):
    # 直接在0～255尺度计算，所以标准差使用15/25/50，无需再除以255。
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=float(sigma), size=clean.shape)
    return np.rint(np.clip(clean.astype(np.float64) + noise, 0, 255)).astype(np.uint8)


def generate(job):
    config = job["config"]
    split = config["split"]
    # 先写配置；若中断，重跑能确认剩余文件使用的是同一配置。
    write_text_safely(job["config_path"],
                      (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    by_sample = {}
    for row in job["rows"]:
        by_sample.setdefault(row["sample_id"], []).append(row)
    created = skipped = 0
    for index, gt_path in enumerate(job["images"], 1):
        pending = []
        for row in by_sample[gt_path.stem]:
            target = DATA / row["noise_path"]
            if target.is_file() and target.stat().st_size > 0:
                skipped += 1
            else:
                pending.append((row, target))
        if pending:
            with Image.open(gt_path) as image:
                expected_size = (config["patch_size"], config["patch_size"])
                if image.mode != "RGB" or image.size != expected_size:
                    raise ValueError(f"要求RGB、尺寸{expected_size}，实际{image.mode}/{image.size}：{gt_path}")
                clean = np.array(image, dtype=np.uint8)
            for row, target in pending:
                result = add_gaussian_noise(clean, row["sigma"], row["noise_seed"])
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_suffix(".png.part")
                Image.fromarray(result).save(temp, format="PNG", compress_level=1)
                temp.replace(target)
                created += 1
        if index % 100 == 0 or index == len(job["images"]):
            print(f"DIV2K {split}：{index}/{len(job['images'])} 张GT", flush=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(job["rows"])
    write_text_safely(job["manifest_path"], buffer.getvalue().encode("utf-8-sig"))
    counts = Counter(row["sigma"] for row in job["rows"])
    print(f"{split} 完成：新增{created}张，跳过{skipped}张；各sigma数量：{dict(sorted(counts.items()))}")
    print(f"对应清单：{job['manifest_path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    original_groups = read_original_splits()
    selected = ("train", "val", "test") if args.split == "all" else (args.split,)
    # 所有选中的GT目录与配置先通过检查，再开始写入Noisy，防止测试集尚未裁剪时部分生成。
    jobs = [build_job(split, original_groups[split], args.seed) for split in selected]
    for job in jobs:
        generate(job)
    print("DIV2K 高斯噪声生成完成。")


if __name__ == "__main__":
    main()
