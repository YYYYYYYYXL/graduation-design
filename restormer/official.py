"""定位项目内固定版本的官方Restormer源码，不依赖pip安装的BasicSR。"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "third_party" / "Restormer"


def use_official_source():
    if not (OFFICIAL_ROOT / "basicsr" / "train.py").is_file():
        raise FileNotFoundError(f"官方训练源码缺失：{OFFICIAL_ROOT}")
    if str(OFFICIAL_ROOT) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_ROOT))
    imported = sys.modules.get("basicsr")
    if imported is not None and Path(imported.__file__).resolve().parent != OFFICIAL_ROOT / "basicsr":
        raise RuntimeError("其他BasicSR已经导入，请在独立进程中启动本项目训练入口")
    return OFFICIAL_ROOT
