from pathlib import Path
import argparse
import csv
import io

data = Path(__file__).resolve().parent / "data"

# 每个 SIDD 场景选多少个目录进入测试集，共16个目录作为测试集
test_counts = {
    "001": 3, "002": 1, "003": 1, "004": 1, "005": 2,
    "006": 2, "007": 2, "008": 2, "009": 1, "010": 1,
}


def select_evenly(items, count) -> set:
    """ 从 items 中均匀选择 count 个元素。items 必须可排序。返回被选为测试集的目录集合 """
    items = sorted(items)
    return {
        items[(2 * i + 1) * len(items) // (2 * count)]
        for i in range(count)
    }


def save_csv(filename, columns, rows, overwrite=False):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8-sig")
    target = data / filename
    if target.exists():
        if target.read_bytes() == content:
            print(f"{filename} 与已有结果一致，保留原文件。")
            return
        if not overwrite:
            raise ValueError(f"{filename} 与已有划分不同；先备份，再显式使用 --overwrite。")
    temp = target.with_suffix(".csv.part")
    temp.write_bytes(content)
    temp.replace(target)

# 划分SIDD数据，给出划分好的数据集名称以及路径
def build_sidd_rows() -> list:
    # SIDD：一个目录有 2 对大图，按目录划分。保持既有选择规则。
    sidd_root = data / "SIDD_Medium_Srgb" / "Data"
    #sidd_dirs是SIDD_Medium_Srgb/Data下的所有子目录组成的列表
    sidd_dirs = sorted(p for p in sidd_root.iterdir() if p.is_dir())
    sidd_rows = []
    # scene是SIDD场景编号，count是测试机需要的数据集
    for scene, count in test_counts.items():
        # p是根据场景编号遍历每个Data下的目录
        folders = [p for p in sidd_dirs if p.name.split("_")[1] == scene]
        #folders长相[每个场景目录的绝对路径（Path列表）]
        if len(folders) < count:
            raise ValueError(f"SIDD 场景 {scene} 的实例不足以满足测试配额。")
        test_dirs = select_evenly(folders, count)
        for folder in folders:#遍历场景目录，被挑中的就是测试集
            split = "test" if folder in test_dirs else "train"
            sidd_rows.append((split, folder.relative_to(data).as_posix()))#as是Path->str,还有反斜杠

            """
            sidd_rows的长相示例：
            [("train","SIDD_Medium_Srgb/Data/0200_010_GP_01600_03200_5500_N"),
            ("test","SIDD_Medium_Srgb/Data/0005_001_S6_00100_00060_4400_N")]
            """

    if len(sidd_dirs) != 160 or sum(row[0] == "train" for row in sidd_rows) != 144:
        raise ValueError("SIDD 必须包含 160 个实例，并划为 144 个训练实例和 16 个测试实例。")
    sidd_rows.sort(key=lambda row: ({"train": 0, "test": 1}[row[0]], row[1]))
    #train,test临时换成01再sort排序
    sidd_rows.append(("val", "sidd_val"))
    return sidd_rows #返回一个列表，内容是[训练集/测试集，路径]

def png_files(folder):#找出folder里的png文件并返回文件路径列表
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png")#后缀转小写


def build_div2k_rows() -> list:
    images = png_files(data / "DIV2K_train_HR") #图片目录列表
    test_root = data / "DIV2K_test_HR"
    val_images = png_files(data / "DIV2K_valid_HR") #验证集图片目录列表
    if test_root.exists(): #如果路径存在就做
        train_images = images
        test_images = png_files(test_root)
    else:
        if len(images) != 800:
            raise ValueError(f"没有独立测试目录时，DIV2K_train_HR 必须有800张，实际{len(images)}张。")
        test_images = sorted(select_evenly(images, 50))
        selected = set(test_images)
        train_images = [p for p in images if p not in selected]
        # 检查划分后的数据，例如真正划分的train_images和预想中的呃750是否相等
    groups = {"train": train_images, "test": test_images, "val": val_images}
    for split, expected in (("train", 750), ("test", 50), ("val", 100)):
        if len(groups[split]) != expected:
            raise ValueError(f"DIV2K {split} 应为{expected}张，实际{len(groups[split])}张。")
    ids = {split: {p.stem for p in files} for split, files in groups.items()}
    """ ids长相示例
    ids = {
        "train": {"0001", "0002", "0003"},
        "test":  {"0009", "0025"},
        "val":   {"0801", "0802"},
    } 
    """
    for a, b in (("train", "test"), ("train", "val"), ("test", "val")):
        if ids[a] & ids[b]: #查看是否有交集
            raise ValueError(f"DIV2K {a}/{b} 原图编号交叉：{sorted(ids[a] & ids[b])}")
    if ids["train"] | ids["test"] != {f"{i:04d}" for i in range(1, 801)}:
        raise ValueError("DIV2K 训练/测试原图应完整覆盖0001～0800，不能缺失、重复或含额外编号。")
    if ids["val"] != {f"{i:04d}" for i in range(801, 901)}:
        raise ValueError("DIV2K 验证原图应完整覆盖0801～0900。")
    return [(split, image.relative_to(data).as_posix())
            for split in ("train", "test", "val") for image in groups[split]]

def main():
    parser = argparse.ArgumentParser(description="生成SIDD/DIV2K划分清单，不移动或修改原始图片。")
    parser.add_argument("--dataset", choices=("all", "sidd", "div2k"), default="all")
    parser.add_argument("--check-only", action="store_true", help="只检查源数据与预期划分，不写CSV")
    parser.add_argument("--overwrite", action="store_true", help="允许更换已有划分；使用前应备份清单及对应输出")
    args = parser.parse_args()
    # 所有选中数据集先检查，再写清单，避免数量异常时仍产生成功结果。
    jobs = [] #保存待输出的清单及其记录
    if args.dataset in ("all", "sidd"):
        jobs.append(("sidd_split.csv", ["split", "folder"], build_sidd_rows()))
    if args.dataset in ("all", "div2k"):
        jobs.append(("div2k_split.csv", ["split", "image"], build_div2k_rows()))
    for filename, columns, rows in jobs:
        print(filename)
        for split in ("train", "test", "val"):
            print(f"  {split}: {sum(row[0] == split for row in rows)} 条")
        if not args.check_only:
            save_csv(filename, columns, rows, overwrite=args.overwrite)
    if args.check_only:
        print("只读检查通过，未写入清单。")


if __name__ == "__main__":
    main()




    
