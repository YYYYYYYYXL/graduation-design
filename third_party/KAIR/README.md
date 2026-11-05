# KAIR DnCNN 推理适配

来源：https://github.com/cszn/KAIR

权重：https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth

参照 `models/network_dncnn.py` 和 `main_test_dncnn.py` 的官方推理配置：RGB、20 层、64 通道、Conv + ReLU，发布权重已合并 BatchNorm，预测噪声残差。项目适配文件为 `dncnn/pretrained.py`，与原有带 BatchNorm 的自训网络分开加载，严格匹配参数，不强行忽略不兼容参数。

许可证保留于本目录 LICENSE。下载文件放在 `checkpoints/dncnn_official/dncnn_color_blind.pth`，不声称其为本项目 SIDD 训练成果。
