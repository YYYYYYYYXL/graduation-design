"""独立进程试测batch，记录速度与显存；仅用训练/验证集，不保存或修改模型。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def choose_trial(rows):
    safe = [row for row in rows if row["status"] == "ok" and row["safe_headroom"]]
    if not safe:
        raise RuntimeError("没有留足显存余量的候选值，请减小batch后重新试测")
    return max(safe, key=lambda row: row["images_per_second"]), max(row["batch"] for row in safe)


def run_trial(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from paired_dataset import DenoisingDataset
    from restormer.model import create_model
    from training_runtime import configure_runtime, loader_options
    from denoising_metrics import calculate_psnr, calculate_ssim, to_uint8
    settings = json.loads(args.profile.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("试测需要有卡模式")
    gpu = torch.cuda.get_device_properties(0)
    if settings.get("gpu_family") and settings["gpu_family"] not in gpu.name:
        raise RuntimeError(f"本配置要求{settings['gpu_family']}，当前是{gpu.name}")
    required = settings.get("minimum_cuda")
    if required and tuple(map(int, (torch.version.cuda or "0.0").split(".")[:2])) < tuple(map(int, required.split("."))):
        raise RuntimeError("5090需要CUDA 12.8或更高的PyTorch构建，请运行setup_remote.sh")
    torch.manual_seed(20261022)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    configure_runtime(settings["performance"], device)
    split = "train" if args.phase == "train" else "val"
    dataset = DenoisingDataset(ROOT / "data", args.dataset, split)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, drop_last=True,
                        **loader_options(args.workers, device, settings["performance"]))
    if len(loader) < args.warmup + args.steps:
        raise ValueError("试测步数超过数据集可用批次，请减小steps或batch")
    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    if args.checkpoint and args.checkpoint.is_file():
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if state["config"]["dataset"] != args.dataset:
            raise ValueError("试测断点数据集不符")
        model.load_state_dict(state["params"], strict=True)
        if args.phase == "train":
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
        del state
    model.train(args.phase == "train")
    minimum_free = gpu.total_memory
    began = None
    for index, (noisy, target, _) in enumerate(loader):
        if index == args.warmup:
            if args.phase == "train" and len(optimizer.state) != len(list(model.parameters())):
                raise RuntimeError("AMP尚未完成实际优化器更新，不能用未分配完整状态的显存值选batch；增加warmup或指定last断点")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
        noisy = noisy.to(device, non_blocking=True)
        if args.phase == "train":
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(noisy)
                loss = F.l1_loss(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError("试测损失非有限")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                prediction = model(noisy)
                if not torch.isfinite(prediction).all():
                    raise RuntimeError("试测输出非有限")
                for output, gt in zip(prediction, target):
                    output, gt = to_uint8(output), to_uint8(gt)
                    calculate_psnr(output, gt)
                    calculate_ssim(output, gt)
        torch.cuda.synchronize()
        minimum_free = min(minimum_free, torch.cuda.mem_get_info()[0])
        if index + 1 == args.warmup + args.steps:
            break
    seconds = time.perf_counter() - began
    peak = torch.cuda.max_memory_reserved()
    return dict(status="ok", phase=args.phase, batch=args.batch, gpu=gpu.name,
                seconds=seconds, measured_images=args.batch*args.steps,
                images_per_second=args.batch*args.steps/seconds,
                peak_reserved_mib=peak/2**20, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                minimum_free_mib=minimum_free/2**20,
                safe_headroom=minimum_free >= 1024*2**20 and peak <= gpu.total_memory*0.95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sidd", "div2k"), default="sidd")
    parser.add_argument("--profile", type=Path, default=ROOT / "options/autodl_5090.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--train-batches", type=int, nargs="+", default=[16,18,20,22,24,28,32])
    parser.add_argument("--val-batches", type=int, nargs="+", default=[4,8,12,16,24,32,48])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--trial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("train", "val"), help=argparse.SUPPRESS)
    parser.add_argument("--batch", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.profile = args.profile if args.profile.is_absolute() else ROOT / args.profile
    settings = json.loads(args.profile.read_text(encoding="utf-8"))
    args.workers = settings["workers"] if args.workers is None else args.workers
    if min(args.warmup,args.steps,*args.train_batches,*args.val_batches)<1 or args.workers<0:
        parser.error("warmup、steps、batch须为正数，workers不能为负")
    args.checkpoint = args.checkpoint or ROOT / "checkpoints/restormer_batch16_20261102" / f"restormer_{args.dataset}_fast_4090d/last.pth"
    if not args.checkpoint.is_absolute():
        args.checkpoint = ROOT / args.checkpoint
    if args.trial:
        import torch
        try:
            result = run_trial(args)
        except torch.OutOfMemoryError:
            result = dict(status="oom",phase=args.phase,batch=args.batch)
        print("PROBE_RESULT="+json.dumps(result),flush=True)
        return
    if args.plan_only:
        print(json.dumps(dict(dataset=args.dataset,train_batches=args.train_batches,val_batches=args.val_batches,
                              checkpoint=str(args.checkpoint),workers=args.workers),ensure_ascii=False,indent=2))
        return
    destination = args.output_dir or ROOT / "results" / f"codex_batch_probe_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
    if not destination.is_absolute(): destination = ROOT / destination
    destination.mkdir(parents=True,exist_ok=False)
    results=[]
    chosen={}
    for phase,batches in (("train",args.train_batches),("val",args.val_batches)):
        for batch in sorted(set(batches)):
            command=[sys.executable,"-B",str(Path(__file__).resolve()),"--trial","--phase",phase,"--batch",str(batch),
                     "--dataset",args.dataset,"--profile",str(args.profile),"--checkpoint",str(args.checkpoint),
                     "--workers",str(args.workers),"--warmup",str(args.warmup),"--steps",str(args.steps)]
            env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
            if "PYTORCH_ALLOC_CONF" not in env:env.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
            print(f"试测{phase} batch={batch}，仅运行临时模型，不改动断点…",flush=True)
            run=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding="utf-8",timeout=300)
            records=[x for x in run.stdout.splitlines() if x.startswith("PROBE_RESULT=")]
            if run.returncode or not records:
                (destination/"codex_failed_trial.log").write_text(run.stdout+run.stderr,encoding="utf-8")
                raise RuntimeError(f"试测异常，已停止；详见{destination}/codex_failed_trial.log")
            row=json.loads(records[-1].split("=",1)[1])
            results.append(row)
            (destination/"codex_trials.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
            print(json.dumps(row,ensure_ascii=False),flush=True)
            if row["status"]=="oom":break
        best,largest=choose_trial([r for r in results if r["phase"]==phase])
        chosen[phase]=dict(recommended_batch=best["batch"],largest_safe_tested_batch=largest)
    settings["models"]["restormer"].update(batch_size=chosen["train"]["recommended_batch"],val_batch_size=chosen["val"]["recommended_batch"])
    settings["workers"]=args.workers
    settings["probe"]=dict(dataset=args.dataset,chosen=chosen,
                          scope="短时间实测推荐值；未覆盖所有整数batch，不能视为绝对上限；正式训练须观察至少两轮及尾批次")
    path=destination/"codex_tuned_profile.json"
    path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"试测完成：{path}。使用该profile启动正式训练；不会自动启动。",flush=True)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
