"""把迁移断点复制到新显卡的输出目录，保留旧断点；不启动训练。"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prepare(source, output_root, profile, dataset):
    settings = json.loads(profile.read_text(encoding="utf-8"))
    tag = settings.get("run_tag", "fast_4090d")
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("不安全的run_tag")
    destination = output_root / f"restormer_{dataset}_{tag}"
    files = ("best.pth", "last.pth", "config.json", "history.csv")
    if destination.resolve() == source.resolve():
        raise ValueError("迁移源和目标目录不能相同")
    if destination.exists():
        raise FileExistsError(f"目标已存在：{destination}。已有迁移结果时直接使用训练入口--resume，不重复复制。")
    for name in files:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config["dataset"] != dataset:
        raise ValueError("断点数据集与目标不符")
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"上次复制的临时目录仍在，请先检查：{temporary}")
    temporary.mkdir(parents=True)
    for name in files:
        shutil.copy2(source / name, temporary / name)
    temporary.replace(destination)
    print(f"已复制断点：{source} -> {destination}；原best/last保持完整。", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("div2k", "sidd"), default="sidd")
    parser.add_argument("--profile", type=Path, default=ROOT / "options/autodl_5090.json")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    source = args.source or ROOT / "checkpoints/restormer_batch16_20261102" / f"restormer_{args.dataset}_fast_4090d"
    profile = args.profile if args.profile.is_absolute() else ROOT / args.profile
    source = source if source.is_absolute() else ROOT / source
    output = args.output_root or Path(json.loads(profile.read_text(encoding="utf-8")).get("output_root", "checkpoints"))
    output = output if output.is_absolute() else ROOT / output
    prepare(source.resolve(), output.resolve(), profile.resolve(), args.dataset)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
