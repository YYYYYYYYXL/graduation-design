"""训练运行选项；快速模式显式启用，默认保留原有复现设置。"""

import torch


def configure_runtime(performance, device):
    fast = performance == "fast" and device.type == "cuda"
    torch.backends.cudnn.benchmark = fast
    torch.backends.cudnn.deterministic = not fast
    if fast:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("快速模式：cuDNN autotune、TF32；不保证逐位复现。", flush=True)


def loader_options(workers, device, performance):
    options = dict(num_workers=workers, pin_memory=device.type == "cuda")
    if workers > 0 and performance == "fast":
        options.update(persistent_workers=True, prefetch_factor=2)
    return options
