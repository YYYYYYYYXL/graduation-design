"""单张图片或两个测试集的 BM3D 去噪，不需要训练。

安装依赖：python -m pip install bm3d==4.0.3
使用示例：python denoise_bm3d.py --input noisy.png --output result.png --sigma 25
两个测试集：python denoise_bm3d.py --dataset all --output results/bm3d
只处理 DIV2K：python denoise_bm3d.py --dataset div2k --output results/bm3d
只处理 SIDD：python denoise_bm3d.py --dataset sidd --output results/bm3d
DIV2K 指定已知 sigma；不指定时仅从噪声图估计，适用于 SIDD 的近似基线。
耗时包含噪声估计和 BM3D，不包含读图、输入转换、输出转换及保存。
批量输出 timing.csv（逐张耗时）和 timing_summary.csv（各组总计与平均耗时）。
"""

import argparse
import csv
import time
from pathlib import Path

import bm3d
import numpy as np
from PIL import Image
from skimage.restoration import estimate_sigma


def denoise(input_path, output_path, sigma=None, *, verbose=True):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"输出已存在，请换一个文件名：{output_path}")
    if output_path.suffix.lower() != ".png":
        raise ValueError("输出文件请使用 .png 扩展名")

    # 读取 RGB 图片，将 0~255 像素转换到 0~1。
    with Image.open(input_path) as image:
        if image.mode != "RGB":
            raise ValueError("输入图片必须是 RGB")
        noisy = np.array(image, dtype=np.float32) / 255.0

    profile = bm3d.BM3DProfile()
    profile.num_threads = 4
    started = time.perf_counter()
    # 没有已知噪声强度时，从噪声图估计；不使用 GT。
    if sigma is None:
        sigma = float(estimate_sigma(noisy, channel_axis=-1, average_sigmas=True)) * 255.0
    if not np.isfinite(sigma) or not 0 <= sigma <= 255:
        raise ValueError("sigma 必须在 0~255 之间")

    # 彩色 BM3D，噪声标准差也要转换到 0~1 尺度。
    clean = bm3d.bm3d_rgb(noisy, sigma_psd=sigma / 255.0, profile=profile) if sigma > 0 else noisy
    elapsed = time.perf_counter() - started

    # 转回 8-bit PNG 并保存，不覆盖已有文件。
    pixels = np.rint(np.clip(clean, 0, 1) * 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as file:
        Image.fromarray(pixels).save(file, format="PNG")
    if verbose:
        print(f"去噪完成，sigma={sigma:.2f}，耗时={elapsed:.4f} 秒，输出：{output_path}", flush=True)
    return float(sigma), elapsed


def denoise_testsets(dataset, data_root, output_root):
    """仅读取测试集噪声图；按数据集及噪声档位分别保存，保持原文件名。"""
    data_root, output_root = Path(data_root).resolve(), Path(output_root).resolve()
    groups = []
    if dataset in ("all", "div2k"):
        for sigma in (15, 25, 50):
            relative = Path("div2k_test") / f"sigma_{sigma}"
            source = data_root / "div2k_test" / "noise" / f"sigma_{sigma}"
            groups.append((source, output_root / relative, sigma))
    if dataset in ("all", "sidd"):
        groups.append((data_root / "sidd_test" / "noisy", output_root / "sidd_test", None))

    for name in ("timing.csv", "timing_summary.csv"):
        if (output_root / name).exists():
            raise FileExistsError(f"计时记录已存在，请换一个 --output 目录：{output_root / name}")

    # 开始前确认所选目录有图片，且不会覆盖数据或已有结果。
    tasks = []
    for source, target, sigma in groups:
        if not source.is_dir():
            raise FileNotFoundError(source)
        if target == data_root or data_root in target.parents:
            raise ValueError("批处理结果请保存在 data 目录之外")
        images = sorted(source.glob("*.png"))
        if not images:
            raise ValueError(f"没有找到测试集 PNG 图片：{source}")
        for image in images:
            destination = target / image.name
            if destination.exists():
                raise FileExistsError(f"输出已存在，请换一个 --output 目录：{destination}")
            tasks.append((image, destination, sigma))

    print(f"开始批量去噪，共 {len(tasks)} 张图片", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    with (output_root / "timing.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["group", "image", "sigma", "denoise_seconds"])
        writer.writeheader()
        for index, (source, target, sigma) in enumerate(tasks, 1):
            print(f"[{index}/{len(tasks)}] ", end="", flush=True)
            actual_sigma, elapsed = denoise(source, target, sigma)
            group = target.parent.relative_to(output_root).as_posix()
            writer.writerow({"group": group, "image": source.name,
                             "sigma": actual_sigma, "denoise_seconds": elapsed})
            file.flush()  # 每张处理后落盘，中途停止仍保留此前的计时记录。
            summary = summaries.setdefault(group, {"count": 0, "total_seconds": 0.0})
            summary["count"] += 1
            summary["total_seconds"] += elapsed

    with (output_root / "timing_summary.csv").open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["group", "count", "total_seconds", "mean_seconds", "threads"])
        writer.writeheader()
        for group, summary in summaries.items():
            mean = summary["total_seconds"] / summary["count"]
            writer.writerow({"group": group, **summary, "mean_seconds": mean, "threads": 4})
            print(f"{group}：{summary['count']} 张，去噪总计 {summary['total_seconds']:.3f} 秒，"
                  f"平均每张 {mean:.4f} 秒", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单张图片或测试集的 BM3D 去噪")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="单张噪声图片路径")
    mode.add_argument("--dataset", choices=("all", "div2k", "sidd"), help="批量处理所选测试集")
    parser.add_argument("--output", required=True, help="单图模式为 PNG 路径；批量模式为结果根目录")
    parser.add_argument("--sigma", type=float, help="8-bit 噪声标准差，如 15、25、50；省略则自动估计")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data",
                        help="批量模式的数据根目录，默认为项目 data")
    args = parser.parse_args()
    if args.dataset:
        if args.sigma is not None:
            parser.error("批量模式自动使用 DIV2K 对应 sigma，SIDD 自动估计，无需传入 --sigma")
        denoise_testsets(args.dataset, args.data_root, args.output)
    else:
        timing_path = Path(args.output).with_suffix(".timing.csv")
        if timing_path.exists():
            raise FileExistsError(f"计时记录已存在：{timing_path}")
        sigma, elapsed = denoise(args.input, args.output, args.sigma)
        with timing_path.open("x", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["image", "sigma", "denoise_seconds"])
            writer.writeheader()
            writer.writerow({"image": Path(args.input).name, "sigma": sigma, "denoise_seconds": elapsed})
