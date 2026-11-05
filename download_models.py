"""只下载官方成品权重，已有文件直接复用，不启动训练。"""

import argparse
from pathlib import Path
from urllib.request import urlopen
import shutil

ROOT = Path(__file__).resolve().parent
DOWNLOADS = {
    "dncnn": ("https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth",
              "checkpoints/dncnn_official/dncnn_color_blind.pth"),
    "restormer-blind": ("https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth",
                        "checkpoints/restormer_official/gaussian_color_denoising_blind.pth"),
    "restormer-real": ("https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth",
                       "checkpoints/restormer_official/real_denoising.pth"),
    "restormer-25": ("https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_sigma25.pth",
                     "checkpoints/restormer_official/gaussian_color_denoising_sigma25.pth"),
}


def download(name):
    url, relative = DOWNLOADS[name]
    target = ROOT / relative
    if target.exists():
        print(f"复用已有文件：{relative}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urlopen(url, timeout=60) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    partial.rename(target)
    print(f"已下载：{relative}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *DOWNLOADS], default="all")
    args = parser.parse_args()
    for name in DOWNLOADS if args.model == "all" else [args.model]:
        download(name)
