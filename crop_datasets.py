from pathlib import Path
import argparse
import csv
import random
import zlib

from PIL import Image

DATA = Path(__file__).resolve().parent / "data"  # 项目的 data 目录
PNG_OPTIONS = {"compress_level": 1}  # 加快 PNG 保存速度


def read_split(filename, split):
    """从 CSV 读取指定集合的记录，返回文件或实例目录的绝对路径列表。"""
    paths = []
    with (DATA / filename).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] == split:
                paths.append(DATA / row[1])
    return paths


def make_dir(path):
    # 创建指定目录和缺少的父目录；目录已存在时直接沿用。
    path.mkdir(parents=True, exist_ok=True)
    return path


def completed(path):
    # 文件存在且非空；这里不检查图片内容是否完整。
    return path.is_file() and path.stat().st_size > 0

def save_png(image, target):
    # 原子替换：中断时不把未保存完整的PNG误当作已完成文件。
    temp = target.with_suffix(".png.part")
    image.save(temp, format="PNG", **PNG_OPTIONS)
    temp.replace(target)


def validate_div2k_split():
    """裁剪前检查 CSV 中的原图数量、编号、文件及目录归属。"""
    expected_counts = {"train": 750, "test": 50, "val": 100}
    groups = {}
    for split in expected_counts:
        groups[split] = read_split("div2k_split.csv", split)

    ids = {}
    for split, paths in groups.items():
        # 数量必须符合方案，同一组中不能重复记录同一个路径。
        if len(paths) != expected_counts[split] or len(set(paths)) != len(paths):
            raise ValueError(f"DIV2K {split} 清单应有{expected_counts[split]}个不重复原图，实际{len(paths)}条。")

        image_ids = set()
        for path in paths:
            if not completed(path):
                raise FileNotFoundError(f"原图不存在或为空：{path}")
            image_ids.add(path.stem)

        # 不同目录中也可能出现同名图片，所以还要检查原图编号。
        if len(image_ids) != len(paths):
            raise ValueError(f"DIV2K {split} 原图编号重复。")
        ids[split] = image_ids

    # 三组两两检查，不能含有相同的原图编号。
    for a, b in (("train", "test"), ("train", "val"), ("test", "val")):
        if ids[a] & ids[b]:
            raise ValueError(f"DIV2K {a}/{b} 原图交叉。")

    expected_train_test_ids = set()
    for number in range(1, 801):
        expected_train_test_ids.add(f"{number:04d}")
    if ids["train"] | ids["test"] != expected_train_test_ids:
        raise ValueError("DIV2K 训练/测试必须完整覆盖0001～0800。")

    expected_val_ids = set()
    for number in range(801, 901):
        expected_val_ids.add(f"{number:04d}")
    if ids["val"] != expected_val_ids:
        raise ValueError("DIV2K 验证编号不符。")

    # 已经分目录存放原图时，CSV 必须沿用这些目录的归属。
    if (DATA / "DIV2K_test_HR").is_dir():
        source_folders = {
            "train": "DIV2K_train_HR",
            "test": "DIV2K_test_HR",
            "val": "DIV2K_valid_HR",
        }
        for split, folder in source_folders.items():
            expected_folder = DATA / folder
            for path in groups[split]:
                if path.parent != expected_folder:
                    raise ValueError(f"DIV2K {split} 应直接使用已有 {folder}/ 原图，不应重新划分。")


def reject_stale_outputs(folder, expected_names):
    """发现不属于当前划分的旧 PNG 时停止，避免混用新旧结果。"""
    existing_names = set()
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() == ".png":
            existing_names.add(path.name)

    unexpected = existing_names - expected_names
    if unexpected:
        raise ValueError(f"{folder} 有不属于当前划分的旧图片，请先备份：{sorted(unexpected)[:5]}")


def rng_for(path):
    # 每张原图固定自己的随机结果，中断后重跑不会改变裁剪位置
    relative_path = path.relative_to(DATA).as_posix()
    seed = zlib.crc32(relative_path.encode("utf-8"))
    return random.Random(seed)


def crop_xy(size, patch_size, rows, cols, index, rng=None):
    """把原图划成 rows×cols 个区域，每个区域提取一个 Patch。"""
    width, height = size
    # 小图块按从左到右、从上到下编号，index 从 0 开始。
    row = index // cols
    col = index % cols
    # left/right 是裁剪框左上角允许的最小/最大 x，top/bottom 对应 y。
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
    # 每处理 10 项更新一次，最后一项也要显示。
    if index % 10 != 0 and index != total:
        return

    line_end = ""
    if index == total:
        line_end = "\n"
    print(f"\r{name}：{index}/{total}", end=line_end, flush=True)


def prepare_sidd(split, patch_size, rows, cols, use_augmentation):
    """读取指定集合的原图对，同步裁剪、增强，再分别保存 Noisy 和 GT。"""
    # 1. 从 CSV 读取当前集合的实例目录，找到每张噪声图对应的 GT。
    folders = read_split("sidd_split.csv", split)
    pairs = []
    for folder in folders:
        noisy_paths = sorted(folder.glob("*_NOISY_SRGB_*.PNG"))
        for noisy_path in noisy_paths:
            gt_name = noisy_path.name.replace("_NOISY_SRGB_", "_GT_SRGB_")
            gt_path = folder / gt_name
            pairs.append((noisy_path, gt_path))  # 每对按“噪声图、GT”的顺序存放。

    # 2. 准备输出目录，计算每对原图需要裁剪的小图块数量。
    base = DATA / f"sidd_{split}"
    noisy_out = make_dir(base / "noisy")
    gt_out = make_dir(base / "gt")
    patch_count = rows * cols

    for n, pair in enumerate(pairs, start=1):
        noisy_path = pair[0]
        gt_path = pair[1]

        # 3. 保留原来的命名规则：实例目录名 + 拍摄编号 + 小图块编号。
        name_parts = noisy_path.stem.split("_")
        capture = name_parts[-1]
        instance_name = noisy_path.parent.name
        prefix = f"{instance_name}_{capture}"

        names = []
        for patch_number in range(1, patch_count + 1):
            name = f"{prefix}_p{patch_number:02d}.png"
            names.append(name)

        # 4. 所有输出都已存在且非空时跳过；缺一张就重做这对原图。
        all_done = True
        for name in names:
            if not completed(noisy_out / name):
                all_done = False
                break
            if not completed(gt_out / name):
                all_done = False
                break

        if all_done:
            show_progress(f"SIDD {split}", n, len(pairs))
            continue

        # 5. 每对原图固定随机种子；测试时不把随机数生成器传给裁剪函数。
        rng = rng_for(noisy_path)
        crop_rng = None
        if use_augmentation:
            crop_rng = rng

        with Image.open(noisy_path) as noisy, Image.open(gt_path) as gt:
            for patch_index in range(patch_count):
                name = names[patch_index]

                # 6. Noisy 和 GT 使用同一个裁剪框，保持像素位置对应。
                # crop_xy返回一个包含四个坐标的元组
                """
                (x, y, x + patch_size, y + patch_size)# 左上角x、左上角y、右下角x、右下角y 
                """
                box = crop_xy(noisy.size, patch_size, rows, cols, patch_index, crop_rng)
                noisy_patch = noisy.crop(box)
                gt_patch = gt.crop(box)

                # 7. 训练时，两块图片使用相同的旋转和翻转参数。
                if use_augmentation:
                    rotation = rng.choice((
                        None, Image.Transpose.ROTATE_90,
                        Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
                    ))
                    flip_lr = rng.random() < 0.5
                    flip_ud = rng.random() < 0.5
                    # augment 是用于真正动手裁剪的函数
                    noisy_patch = augment(noisy_patch, rotation, flip_lr, flip_ud)
                    gt_patch = augment(gt_patch, rotation, flip_lr, flip_ud)

                # 8. 对应小图块使用相同文件名，分别放到 noisy 和 gt 目录。
                save_png(noisy_patch, noisy_out / name)
                save_png(gt_patch, gt_out / name)

        show_progress(f"SIDD {split}", n, len(pairs))

    print(f"  输出：{len(pairs) * patch_count} 对，{patch_size}×{patch_size}")


def prepare_div2k(split, patch_size, patch_count, grid_rows, grid_cols, use_augmentation):
    """裁剪 DIV2K 干净原图；训练时增强，人工加噪交给 add_noise.py。"""
    images = read_split("div2k_split.csv", split)
    gt_out = make_dir(DATA / f"div2k_{split}" / "gt")

    # 先为每张原图生成输出文件名；检查旧结果和后续裁剪都复用这些名字。
    names_by_image = {}
    expected_names = set()
    for image_path in images:
        names = []
        for patch_number in range(1, patch_count + 1):
            name = f"{image_path.stem}_p{patch_number:02d}.png"
            names.append(name)
            expected_names.add(name)
        names_by_image[image_path] = names
    reject_stale_outputs(gt_out, expected_names)

    # 每轮处理一张原图。
    for n, image_path in enumerate(images, start=1):
        names = names_by_image[image_path]
        all_done = True
        for name in names:
            if not completed(gt_out / name):
                all_done = False
                break
        if all_done:
            show_progress(f"DIV2K {split}", n, len(images))
            continue

        # 训练时随机选分区、随机裁剪；验证和测试使用固定分区的中心位置。
        rng = rng_for(image_path)
        crop_rng = None
        if use_augmentation:
            cells = rng.sample(range(grid_rows * grid_cols), patch_count)
            crop_rng = rng
        else:
            cells = list(range(patch_count))

        with Image.open(image_path) as image:
            for patch_index in range(patch_count):
                name = names[patch_index]
                cell = cells[patch_index]
                box = crop_xy(image.size, patch_size, grid_rows, grid_cols, cell, crop_rng)
                patch = image.crop(box)

                if use_augmentation:
                    rotation = rng.choice((
                        None, Image.Transpose.ROTATE_90,
                        Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
                    ))
                    flip_lr = rng.random() < 0.5
                    flip_ud = rng.random() < 0.5
                    patch = augment(patch, rotation, flip_lr, flip_ud)

                save_png(patch, gt_out / name)

        show_progress(f"DIV2K {split}", n, len(images))

    print(f"  输出：{len(images) * patch_count} 张干净图，{patch_size}×{patch_size}")


def prepare_div2k_test():
    # 与验证集相同：2×2分区，各区中心固定裁一块256×256，不增强。
    # 50张测试原图生成200张GT，命名为原图编号_p01.png～_p04.png。
    # 原图归属继续读取CSV；输出目录若残留旧整图，由已有检查拒绝混用。
    prepare_div2k("test", 256, 4, 2, 2, False)


def main():
    """读取命令行选项，检查数据，再调用对应的裁剪函数。"""
    parser = argparse.ArgumentParser(description="按现有CSV裁剪，已有完整输出跳过；")
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


if __name__ == "__main__":
    main()
