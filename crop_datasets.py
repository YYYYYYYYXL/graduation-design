from pathlib import Path
import argparse
import csv
import random
import zlib

from PIL import Image

DATA = Path(__file__).resolve().parent / "data"
PNG_OPTIONS = {"compress_level": 1}  # 加快 PNG 保存速度


def read_split(filename, split):
    with (DATA / filename).open(encoding="utf-8-sig", newline="") as f:
        return [
            DATA / row[1]
            for row in csv.reader(f)
            if row and row[0] == split
        ]


def make_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def completed(path):
    return path.is_file() and path.stat().st_size > 0


def save_png(image, target):
    # 原子替换：中断时不把未保存完整的PNG误当作已完成文件。
    temp = target.with_suffix(".png.part")
    image.save(temp, format="PNG", **PNG_OPTIONS)
    temp.replace(target)


def validate_div2k_split():
    expected_counts = {"train": 750, "test": 50, "val": 100}
    groups = {split: read_split("div2k_split.csv", split) for split in expected_counts}
    for split, paths in groups.items():
        if len(paths) != expected_counts[split] or len(set(paths)) != len(paths):
            raise ValueError(f"DIV2K {split} 清单应有{expected_counts[split]}个不重复原图，实际{len(paths)}条。")
        for path in paths:
            if not completed(path):
                raise FileNotFoundError(f"原图不存在或为空：{path}")
        if len({p.stem for p in paths}) != len(paths):
            raise ValueError(f"DIV2K {split} 原图编号重复。")
    ids = {split: {p.stem for p in paths} for split, paths in groups.items()}
    for a, b in (("train", "test"), ("train", "val"), ("test", "val")):
        if ids[a] & ids[b]:
            raise ValueError(f"DIV2K {a}/{b} 原图交叉。")
    if ids["train"] | ids["test"] != {f"{i:04d}" for i in range(1, 801)}:
        raise ValueError("DIV2K 训练/测试必须完整覆盖0001～0800。")
    if ids["val"] != {f"{i:04d}" for i in range(801, 901)}:
        raise ValueError("DIV2K 验证编号不符。")
    if (DATA / "DIV2K_test_HR").is_dir():
        for split, folder in (("train", "DIV2K_train_HR"), ("test", "DIV2K_test_HR"), ("val", "DIV2K_valid_HR")):
            if any(path.parent != DATA / folder for path in groups[split]):
                raise ValueError(f"DIV2K {split} 应直接使用已有 {folder}/ 原图，不应重新划分。")


def reject_stale_outputs(folder, expected_names):
    unexpected = {p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"} - expected_names
    if unexpected:
        raise ValueError(f"{folder} 有不属于当前划分的旧图片，请先备份：{sorted(unexpected)[:5]}")


def rng_for(path):
    # 每张原图固定自己的随机结果，中断后重跑不会改变裁剪位置
    seed = zlib.crc32(path.relative_to(DATA).as_posix().encode("utf-8"))
    return random.Random(seed)


def crop_xy(size, patch_size, rows, cols, index, rng=None):
    """把原图划成 rows×cols 个区域，每个区域提取一个 Patch。"""
    width, height = size
    row, col = divmod(index, cols)
    left = col * width // cols
    right = (col + 1) * width // cols - patch_size
    top = row * height // rows
    bottom = (row + 1) * height // rows - patch_size
    if not 0 <= index < rows * cols or right < left or bottom < top:
        raise ValueError(f"原图 {width}×{height} 无法按 {rows}×{cols} 分区裁剪 {patch_size}×{patch_size}。")

    if rng is None:
        x = (left + right) // 2
        y = (top + bottom) // 2
    else:
        x = rng.randint(left, right)
        y = rng.randint(top, bottom)

    return (x, y, x + patch_size, y + patch_size)


def augment(image, rotation, flip_lr, flip_ud):
    """旋转、翻转；SIDD 的 Noisy 和 GT 使用完全相同的变换。"""
    if rotation:
        image = image.transpose(rotation)
    if flip_lr:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_ud:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return image


def show_progress(name, index, total):
    if index % 10 == 0 or index == total:
        print(f"\r{name}：{index}/{total}", end="\n" if index == total else "", flush=True)


def prepare_sidd(split, patch_size, rows, cols, use_augmentation):
    folders = read_split("sidd_split.csv", split)
    pairs = [
        (noisy, noisy.with_name(noisy.name.replace("_NOISY_SRGB_", "_GT_SRGB_")))
        for folder in folders
        for noisy in sorted(folder.glob("*_NOISY_SRGB_*.PNG"))
    ]

    base = DATA / f"sidd_{split}"
    noisy_out = make_dir(base / "noisy")
    gt_out = make_dir(base / "gt")
    patch_count = rows * cols

    for n, (noisy_path, gt_path) in enumerate(pairs, start=1):
        capture = noisy_path.stem.rsplit("_", 1)[-1]
        prefix = f"{noisy_path.parent.name}_{capture}"
        names = [f"{prefix}_p{i:02d}.png" for i in range(1, patch_count + 1)]

        # 已完成的原图不重复读取；缺一张 Patch 就重做这对原图
        if all(completed(noisy_out / name) and completed(gt_out / name) for name in names):
            show_progress(f"SIDD {split}", n, len(pairs))
            continue

        rng = rng_for(noisy_path)
        with Image.open(noisy_path) as noisy, Image.open(gt_path) as gt:
            noisy.load()
            gt.load()

            for i, name in enumerate(names):
                box = crop_xy(noisy.size, patch_size, rows, cols, i,
                              rng if use_augmentation else None)
                noisy_patch = noisy.crop(box)
                gt_patch = gt.crop(box)

                if use_augmentation:
                    rotation = rng.choice((
                        None, Image.Transpose.ROTATE_90,
                        Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
                    ))
                    flip_lr = rng.random() < 0.5
                    flip_ud = rng.random() < 0.5
                    noisy_patch = augment(noisy_patch, rotation, flip_lr, flip_ud)
                    gt_patch = augment(gt_patch, rotation, flip_lr, flip_ud)

                save_png(noisy_patch, noisy_out / name)
                save_png(gt_patch, gt_out / name)

        show_progress(f"SIDD {split}", n, len(pairs))

    print(f"  输出：{len(pairs) * patch_count} 对，{patch_size}×{patch_size}")


def prepare_div2k(split, patch_size, patch_count, grid_rows, grid_cols, use_augmentation):
    images = read_split("div2k_split.csv", split)
    gt_out = make_dir(DATA / f"div2k_{split}" / "gt")
    reject_stale_outputs(gt_out, {f"{path.stem}_p{i:02d}.png" for path in images for i in range(1, patch_count + 1)})

    for n, image_path in enumerate(images, start=1):
        names = [f"{image_path.stem}_p{i:02d}.png" for i in range(1, patch_count + 1)]
        if all(completed(gt_out / name) for name in names):
            show_progress(f"DIV2K {split}", n, len(images))
            continue

        rng = rng_for(image_path)
        cells = (
            rng.sample(range(grid_rows * grid_cols), patch_count)
            if use_augmentation else list(range(patch_count))
        )

        with Image.open(image_path) as image:
            image.load()

            for i, (name, cell) in enumerate(zip(names, cells)):
                box = crop_xy(image.size, patch_size, grid_rows, grid_cols, cell,
                              rng if use_augmentation else None)
                patch = image.crop(box)

                if use_augmentation:
                    rotation = rng.choice((
                        None, Image.Transpose.ROTATE_90,
                        Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
                    ))
                    patch = augment(patch, rotation,
                                    rng.random() < 0.5, rng.random() < 0.5)

                save_png(patch, gt_out / name)

        show_progress(f"DIV2K {split}", n, len(images))

    print(f"  输出：{len(images) * patch_count} 张干净图，{patch_size}×{patch_size}")


def prepare_div2k_test():
    # 与验证集相同：2×2分区，各区中心固定裁一块256×256，不增强。
    # 50张测试原图生成200张GT，命名为原图编号_p01.png～_p04.png。
    # 原图归属继续读取CSV；输出目录若残留旧整图，由已有检查拒绝混用。
    prepare_div2k("test", 256, 4, 2, 2, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按现有CSV裁剪，已有完整输出跳过；不修改原图、不加噪。")
    parser.add_argument("--dataset", choices=("all", "sidd", "div2k"), default="all")
    parser.add_argument("--split", choices=("all", "train", "test", "val"), default="all")
    args = parser.parse_args()
    if args.dataset in ("all", "div2k"):
        validate_div2k_split()
    # SIDD 官方验证集 data/sidd_val 已提供裁剪图片，不处理。
    if args.dataset in ("all", "sidd"):
        if args.split in ("all", "train"):
            prepare_sidd("train", 128, 3, 6, True)
        if args.split in ("all", "test"):
            prepare_sidd("test", 256, 4, 4, False)
    if args.dataset in ("all", "div2k"):
        if args.split in ("all", "train"):
            prepare_div2k("train", 128, 7, 3, 3, True)
        if args.split in ("all", "val"):
            prepare_div2k("val", 256, 4, 2, 2, False)
        if args.split in ("all", "test"):
            prepare_div2k_test()

    print("\n裁剪和增强完成。DIV2K 人工加噪留给后续步骤。")
