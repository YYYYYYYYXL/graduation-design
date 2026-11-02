"""AutoDL 单卡速度优先入口；所有默认路径相对于本文件，与终端工作目录无关。"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = {"dncnn": "train_dncnn.py", "p2n": "train_p2n_pretrain.py", "restormer": "train_restormer.py"}


def project_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def build_command(model, dataset, settings, args):
    profile = settings["models"][model]
    output = project_path(args.output_root) / f"{model}_{dataset}_fast_4090d"
    command = [sys.executable, "-B", str(PROJECT_ROOT / SCRIPTS[model]),
               "--dataset", dataset, "--data-root", str(project_path(args.data_root)),
               "--output-dir", str(output), "--device", "cuda",
               "--performance", settings["performance"],
               "--batch-size", str(args.batch_size or profile["batch_size"]),
               "--val-batch-size", str(args.val_batch_size or profile["val_batch_size"]),
               "--workers", str(settings["workers"] if args.workers is None else args.workers),
               "--epochs", str(args.epochs or profile["epochs"])]
    if model != "p2n":
        command.extend(["--precision", profile["precision"]])
    if args.resume:
        command.append("--resume")
    if args.allow_batch_size_change:
        command.append("--allow-batch-size-change")
    return command


def check_data(model, dataset, data_root):
    if model == "p2n" and dataset == "div2k":
        from p2n.dataset import DIV2KPretrainDataset
        training = DIV2KPretrainDataset(data_root, "train")
        validation = DIV2KPretrainDataset(data_root, "val")
    else:
        from paired_dataset import DenoisingDataset
        training = DenoisingDataset(data_root, dataset, "train")
        validation = DenoisingDataset(data_root, dataset, "val")
    if dataset == "div2k" and training.source_ids & validation.source_ids:
        raise ValueError("DIV2K 训练和验证存在原图交叉")
    # 仅检查首尾图片的实际解码与形状；路径和配对清单由 Dataset 检查。
    for group in (training, validation):
        group[0]
        group[len(group) - 1]
    print(f"数据就绪：{model}/{dataset} train={len(training)} val={len(validation)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("all", *SCRIPTS))
    parser.add_argument("dataset", choices=("all", "div2k", "sidd"), nargs="?", default="all")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--profile", default="options/autodl_4090d.json")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-batch-size-change", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="检查全部训练/验证配对路径及样本，不训练、不使用 GPU")
    parser.add_argument("--print-command", action="store_true", help="只显示绝对路径训练命令，不训练")
    args = parser.parse_args()
    if args.allow_batch_size_change and (not args.resume or args.model != "restormer"):
        parser.error("allow-batch-size-change 仅支持 Restormer 的 --resume 续训")
    for field in ("batch_size", "val_batch_size", "epochs"):
        value = getattr(args, field)
        if value is not None and value < 1:
            parser.error(f"{field} 必须为正整数")
    if args.workers is not None and args.workers < 0:
        parser.error("workers 不能为负数")
    data_root, output_root = project_path(args.data_root), project_path(args.output_root)
    if output_root == data_root or data_root in output_root.parents:
        parser.error("输出目录不能位于数据目录内")
    settings = json.loads(project_path(args.profile).read_text(encoding="utf-8"))
    models = list(SCRIPTS) if args.model == "all" else [args.model]
    datasets = ["div2k", "sidd"] if args.dataset == "all" else [args.dataset]
    commands = [(model, dataset, build_command(model, dataset, settings, args))
                for model in models for dataset in datasets]
    if args.print_command:
        for _, _, command in commands:
            print(shlex.join(command))
        return
    for model, dataset, command in commands:
        if not (PROJECT_ROOT / SCRIPTS[model]).is_file():
            raise FileNotFoundError(PROJECT_ROOT / SCRIPTS[model])
        check_data(model, dataset, data_root)
        print(shlex.join(command), flush=True)
    if args.check_only:
        print("路径及数据检查完成；未启动训练，GPU 上限未实测。", flush=True)
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("当前无可用 GPU，请先在 AutoDL 控制台以有卡模式开机，再运行同一命令")
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU：{gpu.name}，显存 {gpu.total_memory / 2**30:.1f} GiB", flush=True)
    for model, _, command in commands:
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        if model == "restormer":
            # 24GB卡上batch16对碎片敏感；保留调用者显式指定的分配器设置。
            if "PYTORCH_ALLOC_CONF" not in environment:
                environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            print("Restormer显存分配：" + environment.get("PYTORCH_ALLOC_CONF", environment.get("PYTORCH_CUDA_ALLOC_CONF", "")), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
