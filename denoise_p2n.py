"""P2N 单张/目录去噪，仅接受 256×256 RGB 图片，每张图独立适应，不使用 GT。

python denoise_p2n.py --input noisy.png --output result.png
python denoise_p2n.py --input data/sidd_test/noisy --output results/p2n/sidd_test
python denoise_p2n.py --input data/div2k_test/noise/sigma_25 --output results/p2n/div2k_test/sigma_25
--steps 0：只使用预训练 U-Net；默认 --steps 100 执行完整 P2N。
默认学习率 1e-6 为当前自训权重的保守初始设置；论文 1e-4 可用 --lr 显式指定。
计时包含每张图参数重置、优化器创建、自监督适应及最终推理，排除权重加载和图片读写。
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from p2n.denoise import P2NDenoiser


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_SIZE = (256, 256)


def validate_image(image, path):
    if image.mode != "RGB":
        raise ValueError(f"图片必须是 RGB：{path}")
    if image.size != INPUT_SIZE:
        raise ValueError(
            f"输入图片必须为 256×256 像素，当前为 {image.width}×{image.height}：{path}"
        )


def read_image(path):
    with Image.open(path) as image:
        validate_image(image, path)
        pixels = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


def save_image(image, path):
    pixels = image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    pixels = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        Image.fromarray(pixels).save(stream, format="PNG")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="单张 256×256 RGB 图片或包含此类 PNG 的目录")
    parser.add_argument("--output", type=Path, required=True, help="单图输出 PNG 路径或批处理输出目录")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "checkpoints/p2n_pretrain/best.pth")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--scale-std", type=float, default=0.75, help="RDC 系数分布的标准差，不是噪声 sigma")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    source, output = args.input.resolve(), args.output.resolve()
    if source.is_dir():
        if source == output or source in output.parents:
            parser.error("输出目录必须位于输入目录之外")
        tasks = [(image, output / image.name) for image in sorted(source.glob("*.png"))]
        timing = output / "timing.csv"
    elif source.is_file():
        if output.suffix.lower() != ".png":
            parser.error("单图输出请使用 .png 扩展名")
        tasks = [(source, output)]
        timing = output.with_suffix(".timing.csv")
    else:
        raise FileNotFoundError(source)
    if not tasks:
        raise ValueError(f"未找到 PNG：{source}")
    for path in [timing] + [destination for _, destination in tasks]:
        if path.exists():
            raise FileExistsError(f"输出已存在，请换一个输出位置：{path}")

    # 在加载模型和创建输出前检查全部输入，避免尺寸不符时占用 GPU 或留下部分结果。
    for image, _ in tasks:
        try:
            with Image.open(image) as pixels:
                validate_image(pixels, image)
        except ValueError as error:
            parser.error(str(error))

    denoiser = P2NDenoiser(args.weights, args.device)
    timing.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "steps", "lr", "scale_std", "seed", "device", "denoise_seconds"]
    with timing.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        total = 0.0
        for index, (image, destination) in enumerate(tasks, 1):
            # 文件名决定局部 seed，避免文件处理顺序改变同一张图片的采样。
            digest = hashlib.sha256(f"{args.seed}|{image.name}".encode()).digest()
            seed = int.from_bytes(digest[:8], "big") % 2**63
            result = denoiser.denoise(read_image(image), args.steps, args.lr, args.scale_std, seed)
            save_image(result.image, destination)
            writer.writerow(dict(zip(fields, [image.name, args.steps, args.lr, args.scale_std,
                                              seed, args.device, result.seconds])))
            stream.flush()
            total += result.seconds
            print(f"[{index}/{len(tasks)}] {image.name}，{args.steps} 步，耗时 {result.seconds:.4f} 秒", flush=True)
        print(f"完成 {len(tasks)} 张，去噪总计 {total:.3f} 秒，平均每张 {total / len(tasks):.4f} 秒", flush=True)


if __name__ == "__main__":
    main()
