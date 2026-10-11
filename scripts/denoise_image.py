"""独立单图去噪：python -m scripts.denoise_image --input 图片路径"""

import argparse
import time
from pathlib import Path

from PIL import Image

from src.inference import denoise_image, load_dncnn


def main():
    parser = argparse.ArgumentParser(description="使用最佳 DnCNN 模型去噪")
    parser.add_argument("--input", required=True, type=Path, help="输入图片路径")
    parser.add_argument("--output", type=Path, help="输出 PNG 路径（不填则自动命名）")
    parser.add_argument("--tile", type=int, default=512, help="分块主体边长")
    parser.add_argument("--fp16", action="store_true", help="CUDA 上尝试 FP16（数值可能稍有不同）")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    weights = project_root / "checkpoints" / "dncnn_sidd_best.pth"
    if not weights.is_file():
        parser.error(f"找不到最佳权重：{weights}")
    if not args.input.is_file():
        parser.error(f"找不到输入图片：{args.input}")
    output = args.output or args.input.with_name(args.input.stem + "_dncnn.png")
    if output.resolve() == args.input.resolve():
        parser.error("输出文件不能覆盖原始输入图片")
    t0 = time.perf_counter()
    model = load_dncnn(weights)
    with Image.open(args.input) as image:
        result = denoise_image(image, model, tile_size=args.tile, fp16=args.fp16)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output, compress_level=1)
    print(f"设备：{next(model.parameters()).device}")
    print(f"图像尺寸：{result.size}")
    print(f"总耗时：{time.perf_counter()-t0:.2f} 秒（包括模型加载和保存）")
    print(f"输出：{output}")


if __name__ == "__main__":
    main()
