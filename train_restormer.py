"""启动项目内适配后的官方Restormer训练程序，不加载官方预训练权重。

python train_restormer.py --dataset div2k
python train_restormer.py --dataset sidd
python train_restormer.py --dataset div2k --resume
python train_restormer.py --config options/restormer_div2k.yml

网络、迭代式训练循环、L1、AdamW及两阶段学习率调度来自固定版本官方源码。
训练参数在 options/restormer_*.yml 配置；不再使用之前自写训练器的--epochs等参数。
--resume 只恢复当前实验自己的完整训练状态；不加时从随机初始化开始。
"""

import argparse
import os
import sys
from pathlib import Path

from restormer.official import PROJECT_ROOT, use_official_source


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", choices=("div2k", "sidd"))
    mode.add_argument("--config", type=Path, help="自定义官方格式YAML配置")
    parser.add_argument("--resume", action="store_true", help="恢复此实验最近保存的训练状态")
    args = parser.parse_args()
    config = (args.config.resolve() if args.config else
              PROJECT_ROOT / "options" / f"restormer_{args.dataset}.yml")
    if not config.is_file():
        raise FileNotFoundError(config)
    official_root = use_official_source()
    previous_directory, previous_argv = Path.cwd(), sys.argv
    try:
        os.chdir(PROJECT_ROOT)
        sys.argv = [str(official_root / "basicsr/train.py"), "-opt", str(config), "--launcher", "none"]
        if args.resume:
            sys.argv.append("--resume")
        from basicsr.train import main as official_main
        official_main()
    finally:
        sys.argv = previous_argv
        os.chdir(previous_directory)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。可加--resume恢复最近已保存的官方训练断点。")
