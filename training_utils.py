"""DnCNN / Restormer 共用的早停、验证和原子保存；不包含模型训练逻辑。"""

import csv
import math

import torch
from tqdm import tqdm

from denoising_metrics import to_uint8, calculate_psnr, calculate_ssim


def early_stopping_status(history, *, min_epochs, patience, min_delta, lr_step):
    """从已有验证历史重建计数，使续训与连续训练采用相同判断。

    参考PSNR只在累计提升达到阈值时更新；best.pth仍独立保存任何新高。
    StepLR在第lr_step的整数倍轮次结束后减半，重置计数以观察下一阶段。
    """
    reference_psnr = None
    bad_epochs = 0
    for row in history:
        score = row["val_psnr"]
        gain = score - reference_psnr if reference_psnr is not None else math.inf
        significant = reference_psnr is None or (score > reference_psnr and
                      (gain >= min_delta or math.isclose(gain, min_delta, rel_tol=1e-9, abs_tol=1e-12)))
        if significant:
            reference_psnr, bad_epochs = score, 0
        else:
            bad_epochs += 1
        if row["epoch"] % lr_step == 0:
            bad_epochs = 0
    last_epoch = history[-1]["epoch"] if history else 0
    return dict(enabled=patience > 0, reference_psnr=reference_psnr, bad_epochs=bad_epochs,
                should_stop=bool(patience > 0 and last_epoch >= min_epochs and bad_epochs >= patience))


def save_checkpoint(payload, path):
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_history(history, path):
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


@torch.no_grad()
def validate(model, loader, device, epoch=None):
    model.eval()
    grouped = {}
    description = f"Epoch {epoch} 验证" if epoch is not None else "验证"
    with tqdm(loader, desc=description, unit="batch", dynamic_ncols=True,
              mininterval=0.5) as progress:
        for noisy, gt, groups in progress:
            prediction = model(noisy.to(device, non_blocking=True))
            if not torch.isfinite(prediction).all():
                raise RuntimeError("验证输出非有限，不能保存本轮best")
            for output, target, group in zip(prediction, gt, groups):
                output_pixels, target_pixels = to_uint8(output), to_uint8(target)
                psnr = calculate_psnr(output_pixels, target_pixels)
                ssim = calculate_ssim(output_pixels, target_pixels)
                if not math.isfinite(psnr) or not math.isfinite(ssim):
                    raise RuntimeError("验证指标非有限")
                values = grouped.setdefault(group, dict(count=0, psnr=0.0, ssim=0.0))
                values["count"] += 1
                values["psnr"] += psnr
                values["ssim"] += ssim
            seen = sum(values["count"] for values in grouped.values())
            progress.set_postfix(PSNR=f"{sum(v['psnr'] for v in grouped.values()) / seen:.3f}",
                                 SSIM=f"{sum(v['ssim'] for v in grouped.values()) / seen:.4f}", refresh=False)
    count = sum(values["count"] for values in grouped.values())
    result = {}
    for metric in ("psnr", "ssim"):
        result[f"val_{metric}"] = sum(values[metric] for values in grouped.values()) / count
        for group, values in sorted(grouped.items()):
            result[f"val_{metric}_{group}"] = values[metric] / values["count"]
    return result
