"""生成远程/按显卡代码与共用数据、断点；只复制白名单，不收集凭据或原始大图。"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "远程"
SOURCES = (
    "train_autodl.py", "train_restormer.py", "training_utils.py", "training_runtime.py",
    "paired_dataset.py", "denoising_metrics.py", "run_autodl.sh", "run_5090.sh", "setup_remote.sh",
    "prepare_restormer_resume.py", "probe_restormer.py", "requirements-restormer.txt",
    "restormer/__init__.py", "restormer/model.py", "restormer/LICENSE.MIT",
    "third_party/Restormer/basicsr/models/archs/restormer_arch.py",
    "third_party/Restormer/LICENSE.md", "third_party/Restormer/UPSTREAM.json", "third_party/Restormer/ADAPTATIONS.md",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_archive(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    original = sha256(source)
    if target.exists():
        if sha256(target) != original:
            raise FileExistsError(f"已有同名归档与源文件不同，未覆盖：{target}")
    else:
        temporary = target.with_suffix(target.suffix+".part")
        if temporary.exists():
            raise FileExistsError(temporary)
        shutil.copyfile(source, temporary)
        if sha256(temporary) != original:
            raise RuntimeError(f"复制校验失败：{temporary}")
        temporary.replace(target)
    return dict(path=target.relative_to(DESTINATION).as_posix(), bytes=target.stat().st_size, sha256=original)


def validate_data(path):
    counts = Counter()
    allowed = {f"data/{d}_{s}" for d in ("div2k", "sidd") for s in ("train", "val", "test")}
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["codex_graduation_data_20261102.tar"]:
            raise ValueError("数据ZIP内部不是预期的裁剪数据tar")
        with archive.open(archive.namelist()[0]) as stream, tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                name = member.name
                if ".." in Path(name).parts or name.startswith("/") or member.issym() or member.islnk():
                    raise ValueError(f"归档路径异常：{name}")
                if not member.isfile():
                    continue
                if "/".join(name.split("/")[:2]) not in allowed and name not in ("data/div2k_split.csv", "data/sidd_split.csv"):
                    raise ValueError(f"归档包含非预期数据：{name}")
                counts["/".join(name.split("/")[:3])] += 1
            while stream.read(8*1024*1024):
                pass  # 读到ZIP成员结尾，检查CRC。
    expected = {"data/div2k_train/gt":5250,"data/div2k_train/noise":5250,
                "data/div2k_val/gt":400,"data/div2k_val/noise":1200,
                "data/div2k_test/gt":200,"data/div2k_test/noise":600,
                "data/sidd_train/gt":5184,"data/sidd_train/noisy":5184,
                "data/sidd_val/input_crops":1280,"data/sidd_val/target_crops":1280,
                "data/sidd_test/gt":512,"data/sidd_test/noisy":512}
    for group,count in expected.items():
        if counts[group] != count:
            raise ValueError(f"数据数量异常：{group}={counts[group]}，应为{count}")
    return dict(zip_crc_checked=True, raw_large_images=False, files_by_group=dict(counts))


def validate_checkpoints(path):
    history = {}
    files = {}
    prefix = "checkpoints/restormer_batch16_20261102/"
    with gzip.open(path,"rb") as stream, tarfile.open(fileobj=stream,mode="r|") as archive:
        for member in archive:
            if ".." in Path(member.name).parts or member.name.startswith("/") or member.issym() or member.islnk():
                raise ValueError("断点归档路径异常")
            if member.isfile() and member.name.startswith(prefix):
                files[member.name] = member.size
                if member.name.endswith("history.csv"):
                    with archive.extractfile(member) as f:
                        rows=list(csv.DictReader(io.StringIO(f.read().decode("utf-8-sig"))))
                    dataset = "sidd" if "restormer_sidd_" in member.name else "div2k"
                    history[dataset] = {"last_epoch":int(rows[-1]["epoch"]),
                                        "best_epoch":int(max(rows,key=lambda r:float(r["val_psnr"]))["epoch"])}
        while stream.read(8*1024*1024):pass
    for dataset in ("div2k", "sidd"):
        directory = prefix + f"restormer_{dataset}_fast_4090d/"
        for name in ("best.pth","last.pth","config.json","history.csv"):
            if files.get(directory+name,0)<=0:raise ValueError(f"断点文件缺失：{directory+name}")
    if history != {"div2k":{"last_epoch":75,"best_epoch":13},"sidd":{"last_epoch":33,"best_epoch":16}}:
        raise ValueError(f"断点轮次与本次迁移来源不同：{history}")
    return dict(gzip_crc_checked=True,history=history,files=files)


def build_code(label, source_profile):
    base = DESTINATION / label
    code = base / "代码"
    code.mkdir(parents=True, exist_ok=True)
    names=[]
    for name in SOURCES:
        target=code/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes().replace(b"\r\n",b"\n"))
        names.append(name)
    settings=json.loads((ROOT/source_profile).read_text(encoding="utf-8"))
    settings["models"]={"restormer":settings["models"]["restormer"]}
    if label=="4090D":
        settings.update(run_tag="fast_4090d",output_root="checkpoints/remote_4090d",gpu_family="4090")
    active="options/active_gpu.json"
    for name,value in ((active,settings),("options/autodl_5090.json",json.loads((ROOT/"options/autodl_5090.json").read_text(encoding="utf-8")))):
        target=code/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        names.append(name)
    # train_autodl默认入口也限制为本包的Restormer配置。
    (code/"options/autodl_4090d.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    names.append("options/autodl_4090d.json")
    (base/"参数.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    launcher='''#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$PROJECT_ROOT/run_autodl.sh" restormer --profile "$PROJECT_ROOT/options/active_gpu.json" "$@"
'''
    (code/"run_gpu.sh").write_text(launcher,encoding="utf-8",newline="\n")
    names.append("run_gpu.sh")
    resume='''#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${DENOISE_PYTHON:-/root/venvs/denoise/bin/python}"
TARGET="$PROJECT_ROOT/{output_root}/restormer_sidd_{run_tag}"
if [[ ! -f "$TARGET/last.pth" ]]; then
    "$PYTHON" -B "$PROJECT_ROOT/prepare_restormer_resume.py" --dataset sidd --profile "$PROJECT_ROOT/options/active_gpu.json"
fi
exec bash "$PROJECT_ROOT/run_gpu.sh" sidd --resume --allow-batch-size-change "$@"
'''.replace("{output_root}",settings["output_root"]).replace("{run_tag}",settings["run_tag"])
    (code/"resume_sidd.sh").write_text(resume,encoding="utf-8",newline="\n")
    names.append("resume_sidd.sh")
    (code/"README_REMOTE.md").write_bytes((DESTINATION/"README.md").read_bytes().replace(b"\r\n",b"\n"))
    names.append("README_REMOTE.md")
    package=base/f"远程代码_{label}.zip"
    temporary=package.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in sorted(set(names)):
            archive.write(code/name,"graduation-design/"+name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:raise ValueError("代码ZIP校验失败")
    temporary.replace(package)
    return dict(path=package.relative_to(DESTINATION).as_posix(),bytes=package.stat().st_size,
                sha256=sha256(package),parameters=settings,
                source_files_sha256={name:sha256(code/name) for name in sorted(set(names))})


def build_5090_bundle():
    """把既有代码、嵌套数据tar及最新两组断点展开为单层可部署ZIP。"""
    metadata=json.loads((DESTINATION/"文件校验.json").read_text(encoding="utf-8"))
    code_path=DESTINATION/metadata["5090"]["path"]
    data_path=DESTINATION/metadata["data"]["path"]
    checkpoints_path=DESTINATION/metadata["checkpoints"]["path"]
    for source,key in ((code_path,"5090"),(data_path,"data"),(checkpoints_path,"checkpoints")):
        if source.stat().st_size!=metadata[key]["bytes"] or sha256(source)!=metadata[key]["sha256"]:
            raise ValueError(f"来源归档与已校验记录不一致：{source}")
    package=DESTINATION/"5090/5090专用整包_代码数据断点.zip"
    temporary=package.with_suffix(".zip.part")
    if temporary.exists():raise FileExistsError(f"已有未完成整包，请先检查：{temporary}")
    seen=set()
    counts=Counter()
    def add_file(archive,name,stream,size,compressed=False):
        if name in seen or ".." in Path(name).parts or not name.startswith("graduation-design/"):
            raise ValueError(f"重复或不安全的整包路径：{name}")
        seen.add(name)
        info=zipfile.ZipInfo(name)
        info.file_size=size
        info.compress_type=zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
        info.external_attr=(0o100644 << 16)
        with archive.open(info,"w",force_zip64=True) as target:
            shutil.copyfileobj(stream,target,1024*1024)
    with zipfile.ZipFile(temporary,"w",allowZip64=True,compression=zipfile.ZIP_STORED) as bundle:
        print("整包：写入5090训练代码…",flush=True)
        with zipfile.ZipFile(code_path) as archive:
            for item in archive.infolist():
                if item.is_dir() or item.filename.endswith("/README_REMOTE.md"):continue
                with archive.open(item) as stream:add_file(bundle,item.filename,stream,item.file_size)
                counts["code_files"]+=1
        guide=(DESTINATION/"5090/整包使用说明.md").read_bytes()
        add_file(bundle,"graduation-design/README_REMOTE.md",io.BytesIO(guide),len(guide))
        counts["code_files"]+=1
        print("整包：写入两个数据集的全部裁剪train/val/test…",flush=True)
        with zipfile.ZipFile(data_path) as archive:
            with archive.open("codex_graduation_data_20261102.tar") as stream,tarfile.open(fileobj=stream,mode="r|") as tar:
                for member in tar:
                    if not member.isfile():continue
                    if not member.name.startswith("data/"):raise ValueError("非预期数据路径")
                    with tar.extractfile(member) as source:
                        add_file(bundle,"graduation-design/"+member.name,source,member.size)
                    counts["data_files"]+=1
                    if member.name.endswith(".png"):counts["data_pngs"]+=1
                    if counts["data_files"]%2000==0:print(f"数据已写入{counts['data_files']}个文件",flush=True)
                while stream.read(8*1024*1024):pass
        print("整包：写入最新DIV2K和SIDD的best/last及训练记录…",flush=True)
        prefix="checkpoints/restormer_batch16_20261102/"
        with gzip.open(checkpoints_path,"rb") as stream,tarfile.open(fileobj=stream,mode="r|") as archive:
            for member in archive:
                if not member.isfile() or not member.name.startswith(prefix):continue
                if Path(member.name).name not in ("best.pth","last.pth","config.json","history.csv"):
                    raise ValueError("非预期断点文件")
                with archive.extractfile(member) as source:
                    add_file(bundle,"graduation-design/"+member.name,source,member.size,compressed=member.name.endswith(".pth"))
                counts["checkpoint_files"]+=1
            while stream.read(8*1024*1024):pass
    expected_data=sum(metadata["data"]["validation"]["files_by_group"].values())
    if counts["data_files"]!=expected_data or counts["checkpoint_files"]!=8:
        raise ValueError(f"整包文件数量异常：{counts}")
    print("整包：校验全部文件CRC和目录结构…",flush=True)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:raise ValueError("整包ZIP CRC校验失败")
        active=json.loads(archive.read("graduation-design/options/active_gpu.json").decode("utf-8"))
        if active["run_tag"]!="fast_5090":raise ValueError("整包显卡配置错误")
        if any(name.endswith((".tar",".tar.gz",".zip")) for name in archive.namelist()):
            raise ValueError("整包内不应再嵌套压缩包")
    temporary.replace(package)
    record=dict(path=package.relative_to(DESTINATION).as_posix(),bytes=package.stat().st_size,
                sha256=sha256(package),zip_crc_checked=True,no_nested_archives=True,counts=dict(counts),
                profiles=active,source_archives={k:metadata[k]["sha256"] for k in ("5090","data","checkpoints")})
    (DESTINATION/"5090/整包校验.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"5090整包完成：{package.name}，{package.stat().st_size/1e9:.2f} GB",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-archive",type=Path)
    parser.add_argument("--checkpoints-archive",type=Path)
    parser.add_argument("--bundle-5090-only",action="store_true",help="使用已生成的远程目录，制作单层5090代码+数据+断点整包")
    args=parser.parse_args()
    if args.bundle_5090_only:
        build_5090_bundle()
        return
    if not args.data_archive or not args.checkpoints_archive:
        parser.error("生成分包需要--data-archive和--checkpoints-archive；生成5090整包使用--bundle-5090-only")
    manifest={"scope":"Restormer remote migration, processed data only; RTX5090 not measured yet"}
    print("检查完整裁剪数据与ZIP CRC…",flush=True)
    validation=validate_data(args.data_archive)
    print("检查断点轮次与GZIP CRC…",flush=True)
    checkpoints=validate_checkpoints(args.checkpoints_archive)
    print("复制共用数据与断点，并逐个校验SHA256…",flush=True)
    manifest["data"]=copy_archive(args.data_archive,DESTINATION/"数据/裁剪数据.zip")
    manifest["data"]["validation"]=validation
    manifest["checkpoints"]=copy_archive(args.checkpoints_archive,DESTINATION/"断点/checkpoints.tar.gz")
    manifest["checkpoints"]["validation"]=checkpoints
    manifest["4090D"]=build_code("4090D","options/autodl_4090d.json")
    manifest["5090"]=build_code("5090","options/autodl_5090.json")
    (DESTINATION/"文件校验.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("完成：远程目录，包含两套代码ZIP、共用数据和断点。",flush=True)


if __name__=="__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
