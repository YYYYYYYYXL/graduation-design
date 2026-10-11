# 基于深度学习的图像去噪平台设计与实现

> 毕业设计项目 · 当前阶段：**DnCNN 模型训练、验证及独立推理功能已完成；Web 平台后端与前端尚未开发**  
> 项目现状更新：2026-10-10

## 1. 项目简介

本项目以图像去噪为应用场景，计划设计并实现一个可通过浏览器使用的图像去噪平台。最终目标是让用户上传图像、选择去噪方法、获取处理结果，并对不同算法的效果进行展示与比较。

当前已完成的工作集中在**第一个深度学习去噪模型 DnCNN**：使用 SIDD Medium sRGB 数据构建训练/验证/测试划分，实现基于 PyTorch 的模型训练与断点续训、数据缓存优化、模型指标评估，以及可复用的图像推理和命令行去噪工具。

**项目目前不是已上线的 Web 平台**：FastAPI 后端、前端交互、多算法接入与系统部署属于下一阶段工作，不能将其作为已经实现的功能。

## 2. 技术栈与运行环境

| 类别 | 当前技术 / 配置 | 说明 |
| --- | --- | --- |
| 开发语言 | Python 3.10 | 使用 Conda 环境 `denoise310` |
| 深度学习 | PyTorch 2.2.2+cu121 | 历史实测版本；以本机环境为准 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU（6 GB） | 之前已验证 CUDA 训练和推理可用 |
| 图像处理 | Pillow、NumPy | 图像读取、裁剪、数组转换与缓存 |
| 数据与计算 | CSV、PyTorch Dataset / DataLoader | 划分结果保存为 CSV，模型从 Dataset 读取数据 |
| 模型评价 | PSNR、SSIM、MSE / Val Loss | SSIM 使用 scikit-image |
| 实验环境 | Jupyter Notebook（VS Code） | 数据检查、模型分析和可视化 |
| 计划中的 Web 后端 | FastAPI | **尚未实现**，拟用于上传、调用模型与返回结果 |

`requirements-extra.txt` 列出了项目辅助依赖；已有的 CUDA 版 PyTorch 请继续使用当前 Conda 环境，不要为了安装辅助依赖而随意替换它。

## 3. 项目目录

以下为**代码整理后的目标结构**。其中 `data/`、`checkpoints/`、`.git/`、`.vscode/` 和 `.gitignore` 以本地现有内容为准；原始数据和缓存通常不放入代码压缩包或 Git 仓库。

```text
graduation-design/
├── .git/                           # Git 版本历史（本地已有）
├── .vscode/                        # VS Code 本地配置（如存在）
├── .gitignore                      # Git 忽略规则（保留已有配置）
├── README.md                       # 项目总览、目录、使用说明和计划
├── requirements-extra.txt          # 补充 Python 依赖；不负责重装 PyTorch
│
├── data/                           # 数据与缓存（大型文件通常不提交）
│   ├── SIDD_Medium_Srgb/
│   │   └── Data/                    # SIDD Medium 原始 NOISY / GT 图像
│   ├── DIV2K_train_HR/             # 本地已有的 DIV2K 图像（当前 DnCNN 未使用）
│   ├── DIV2K_valid_HR/             # 本地已有的 DIV2K 图像（当前 DnCNN 未使用）
│   ├── splits/
│   │   └── sidd_split.csv          # 固定的场景级 train/val/test 划分
│   └── cache/
│       ├── sidd_full/              # 222 对训练完整图像的 uint8 .npy 缓存
│       │   ├── test_0000_noisy.npy # 历史命名；实际属于训练集！
│       │   ├── test_0000_gt.npy
│       │   └── ...
│       └── sidd_val/               # 46 对验证图像的 256×256 中心裁剪缓存
│           ├── val_0000_noisy.npy
│           ├── val_0000_gt.npy
│           └── ...
│
├── checkpoints/
│   ├── dncnn_sidd_best.pth         # 最低 Val Loss 对应的 DnCNN 权重
│   ├── dncnn_sidd_last.pth         # 最新轮次的模型、优化器与续训状态
│   └── ...                         # 如有旧轮次备份，以本机实际文件为准
│
├── src/                            # 正式业务与算法源码
│   ├── __init__.py
│   ├── inference.py                 # DnCNN 权重加载、分块去噪推理
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── sidd_dataset.py          # 通用 SIDD 训练/验证/测试数据读取
│   │   └── sidd_train_dataset.py    # 正式训练用完整缓存与随机 Patch 采样
│   ├── models/
│   │   ├── __init__.py
│   │   └── dncnn.py                 # DnCNN 20 层网络及残差去噪实现
│   └── train/
│       ├── __init__.py
│       └── train_dncnn.py          # 正式训练、验证、保存与断点续训入口
│
├── scripts/                        # 独立命令行/数据维护工具
│   ├── __init__.py
│   ├── denoise_image.py             # 输入单张图片、调用模型、保存结果
│   └── prepare_sidd_cache.py       # 缓存只读检查、按需补建/重建
│
├── notebooks/                      # 实验与长期检测 Notebook
│   ├── prepare_sidd_split.ipynb    # SIDD 场景级划分及 CSV 一致性检查
│   ├── sidd_data_test.ipynb        # 原始图片、配对、数据完整性与样本展示
│   └── dncnn_evaluation_test.ipynb # 最佳模型加载、单图展示、完整验证集评价
│
└── docs/                           # 项目技术记录
    ├── DnCNN训练耗时异常与数据加载优化记录_2026-10-08.md
    ├── DnCNN训练性能瓶颈定位与缓存优化记录_2026-10-09.md
    └── 整理迁移说明.md
```

> 说明：`__init__.py` 用于标识 Python 包；`__pycache__/`、`.pyc` 等自动生成内容不属于应维护的项目源码。目录列出的 DIV2K 数据是本地既有资源，不代表已经进入当前模型训练。

## 4. 每个主要文件做什么

### 4.1 `src/`：正式功能模块

| 文件 | 主要职责 |
| --- | --- |
| `src/models/dncnn.py` | 定义 DnCNN 网络：默认输入 RGB 三通道、64 个特征通道、20 个卷积层；使用残差学习预测噪声，再用输入减去预测噪声获得去噪结果。 |
| `src/datasets/sidd_dataset.py` | 从 `sidd_split.csv` 按 `train` / `val` / `test` 读取 NOISY/GT 配对，执行路径检查、RGB 转换、训练随机裁剪/增强或验证/测试中心裁剪；验证集优先使用固定 `.npy` 缓存。 |
| `src/datasets/sidd_train_dataset.py` | 定义正式训练类 `SIDDTrainDataset`；从训练全图 `.npy` 缓存内存映射读取，每对原图随机提取 4 对 128×128 Patch，并同步增强。 |
| `src/train/train_dncnn.py` | 训练入口；构建数据加载器、DnCNN、MSE 损失与 Adam 优化器，完成训练和验证；保存最佳模型与最近一次训练断点。 |
| `src/inference.py` | 提供 `load_dncnn()` 和 `denoise_image()`；从最佳权重加载模型，对任意尺寸 RGB 图像采用带重叠边界的分块推理，降低大图单次显存需求。 |

### 4.2 `scripts/`：可直接运行的程序

| 文件 | 主要职责 |
| --- | --- |
| `scripts/prepare_sidd_cache.py` | 默认**只检查**训练及验证缓存的文件存在性、dtype 和 shape；支持显式 `--build-missing` 补建、`--rebuild` 覆盖重建。 |
| `scripts/denoise_image.py` | 命令行单图去噪工具；指定输入路径、可选输出路径、分块尺寸和 CUDA FP16，默认保存同目录的 `_dncnn.png`。 |

### 4.3 `notebooks/`：长期实验与检查

| Notebook | 主要内容 |
| --- | --- |
| `prepare_sidd_split.ipynb` | 扫描 SIDD Medium 场景实例，按 `scene_id` 固定划分 train/val/test，检查场景交叉、图像配对与现有 CSV 顺序是否一致；**默认不写入/覆盖 CSV**。 |
| `sidd_data_test.ipynb` | 统计 SIDD 原始图像数量、检查 NOISY/GT 配对及尺寸/模式，随机裁剪展示样本；可选完整解码检查，附带 DIV2K 文件数检查。 |
| `dncnn_evaluation_test.ipynb` | 独立加载 `best.pth`，统计模型结构；单图三图对比（NOISY/DnCNN/GT）；验证集 Val Loss、PSNR、SSIM；逐样本改善情况和效果较差样本展示；可选外部图片推理。 |

命名规则：**临时实验或一次性测试文件使用 `test_` 前缀**（如 `test_fp16_speed.ipynb`）；**长期保留、用于检查输出的 Notebook 使用 `_test.ipynb` 后缀**。正式任务按 `train_`、`prepare_` 等职责命名。历史训练缓存 `test_XXXX.npy` 是既有兼容例外，不代表测试集。

### 4.4 `docs/`：技术文档

| 文档 | 记录内容 |
| --- | --- |
| `DnCNN训练耗时异常与数据加载优化记录_2026-10-08.md` | 记录初期约 22 分钟/Epoch 的问题，定位重复 PNG 解码瓶颈，以及分组取样、双 Worker 数据加载优化。 |
| `DnCNN训练性能瓶颈定位与缓存优化记录_2026-10-09.md` | 记录分段计时、完整图像缓存与固定 Patch 池的比较、最终采用完整 `.npy` 缓存、验证集缓存方案及加速效果。 |
| `整理迁移说明.md` | 记录旧 Notebook、Dataset 文件的合并/更名去向，解释重构兼容措施、未改动的权重与数据，以及迁移注意事项。 |
| `README.md`（本文件） | 面向项目使用者的总入口，说明项目目标、目录、文件职责、运行方法、已验证成果及后续开发计划。 |

## 5. 数据集与划分

目前 DnCNN 使用 **SIDD Medium sRGB** 的成对噪声图像（NOISY）与参考干净图像（GT）。数据集总计 **320 对**，按照场景编号划分，而不是按单张图片随机划分：

| 划分 | 图像对数 | 场景编号 | 当前用途 |
| --- | ---: | --- | --- |
| 训练集 `train` | 222 | `001`～`006` | 更新模型权重 |
| 验证集 `val` | 46 | `007`、`010` | 训练期间选择最佳模型，进一步计算 PSNR/SSIM |
| 测试集 `test` | 52 | `008`、`009` | 保留用于模型确定后的独立最终评价 |
| **合计** | **320** | 场景级隔离 | |

`data/splits/sidd_split.csv` 记录划分、场景信息以及每对 NOISY/GT 的相对路径。当前历史 CSV 的 SHA-256：

```text
ada84dbb4af1d2d4a68813022fbf5adcfffc0a7f94894ac803787cbea1f52832
```

**数据注意事项：**

1. 训练全图缓存保留历史 `test_` 文件前缀，实际对应 `train` 样本，切勿因为名称包含 `test` 而删除或误认为测试集。
2. 缓存下标依赖 CSV 原有样本顺序；未经同步核对，不要重新排序或覆盖 CSV。
3. 旧缓存没有 `manifest.json`。现有只读检查可以确认文件存在、形状及类型合理，**不能独立证明每一个缓存索引都与 CSV 图像完全对应**。
4. 训练与验证使用缓存加速；当前测试集从原始 PNG 解码，再固定中心裁剪 256×256。完整原图评价与 256×256 裁剪评价不能混作同一口径。
5. SIDD Small 与 Medium 存在场景实例来源重叠；从 Small 任意选取的图片不能直接宣称为与 Medium 完全独立的外部数据。

## 6. DnCNN 训练设置与已有成果

### 6.1 模型和训练配置

| 项目 | 已使用的配置 |
| --- | --- |
| 网络 | DnCNN，全卷积结构；20 个卷积层，无全连接层 |
| 输入与输出 | RGB，3 通道；模型输出为去噪图像 |
| 训练 Patch | 128×128 |
| 每对训练原图生成 Patch | 4 对 |
| 每 Batch 原图对数 | 2 对，即 8 个 Patch |
| 每轮数据量 | 222 对原图 × 4 = 888 Patch；111 个 Batch |
| 优化器 / 损失函数 | Adam / MSELoss |
| 历史学习率 | 0.001 |
| 已完成轮数 | 20 Epoch |
| 最佳轮次 | 第 17 个 Epoch（按最低验证损失选出） |

### 6.2 最佳模型的验证集结果

以下结果来自**本机对 46 对验证图像的实际运行**，并在代码整理后复核一致；不是对 52 对独立测试集的最终报告。

| 指标 | 原始 NOISY | DnCNN 最佳权重 | 改善 |
| --- | ---: | ---: | ---: |
| 平均 PSNR | 29.33 dB | **32.90 dB** | **+3.57 dB** |
| 平均 SSIM | 0.5811 | **0.7501** | **+0.1690** |
| Val Loss（MSE） | — | **0.000754** | — |

评价口径：使用 46 对固定 256×256 验证裁剪；单张图像分别计算 PSNR/SSIM 后取算术平均；Val Loss 使用未 `clamp` 的模型输出，PSNR/SSIM 使用裁剪到 `[0,1]` 的输出。以上成绩证明**当前验证集上具有去噪收益**，不等于所有手机照片均能达到相同效果。

### 6.3 训练性能优化

初始训练重复解码高分辨率 PNG，单个完整 Epoch 约 22 分钟。通过分组取样、双 Worker、完整图像 `.npy` 缓存和验证集固定裁剪缓存，历史记录中的完整 Epoch 曾达到 **31.77 秒**（特定轮次的实测值，非恒定速度）。详细实验依据见 `docs/` 两份性能记录。

另有一张 5328×3000 的高分辨率图片在早期**整图推理**测试中总耗时 **161.76 秒**。为降低显存占用，目前已实现分块推理代码，但**尚未对所有大图完成统一的正式测速与性能验收**，不能宣称推理速度已达到平台要求。

### 6.4 两类 `.pth` 文件

- `checkpoints/dncnn_sidd_best.pth`：仅保存模型权重；推理、展示和指标评价优先使用。按当前训练记录，其最佳验证损失来自 Epoch 17。
- `checkpoints/dncnn_sidd_last.pth`：保存最新模型、优化器状态、已完成轮次与历史最佳损失；正式续训默认从此文件恢复。当前已核对断点记录为 Epoch 20。

`best` **不会每轮覆盖**，只有新的 Val Loss 低于历史最佳时才更新；`last` 在每个完整 Epoch 结束时更新。二者不可无条件互换。

## 7. 安装与运行

以下命令均假设终端当前位于**项目根目录**，使用 Windows PowerShell 和现有 Conda 环境。

### 7.1 激活环境并安装辅助依赖

```powershell
conda activate denoise310
pip install -r requirements-extra.txt
```

如果当前环境已可正常运行 Notebook 和训练程序，无需重复安装依赖。`requirements-extra.txt` 不包含 PyTorch，请保留已验证可用的 CUDA 版本。

### 7.2 只读检查训练/验证缓存

```powershell
python -m scripts.prepare_sidd_cache
```

默认不覆盖文件。脚本另提供 `--build-missing`（补缺）和 `--rebuild`（覆盖重建），**正常检查时不要加这两个参数**；重新生成缓存前应确认磁盘空间、原始 PNG 与 CSV 的对应关系。

### 7.3 Notebook 数据检查及模型评估

使用 VS Code 打开对应 `.ipynb`，选择 `denoise310` Kernel，依次运行：

1. `notebooks/prepare_sidd_split.ipynb`：检查场景划分和现有 CSV；默认不写入。
2. `notebooks/sidd_data_test.ipynb`：检查原始数据/配对，并查看样本。
3. `notebooks/dncnn_evaluation_test.ipynb`：加载最佳权重，展示单图去噪，对 46 对验证图像计算 Val Loss、PSNR、SSIM。

不要求每次运行全部 Notebook；查看模型效果通常只运行第 3 项。

### 7.4 用已有模型处理一张图片

```powershell
python -m scripts.denoise_image --input "D:\Pictures\noisy.jpg" --tile 512
```

默认读取 `checkpoints/dncnn_sidd_best.pth`，把输出保存在原图旁边，例如 `noisy_dncnn.png`。可选参数：

```powershell
python -m scripts.denoise_image --input "D:\Pictures\noisy.jpg" --output "D:\Pictures\result.png" --tile 512 --fp16
```

`--tile` 为分块核心区域边长；代码额外保留重叠边界以避免明显的拼接接缝。`--fp16` 仅在 CUDA 上尝试混合精度，可能改变少量像素结果，且不保证一定更快。推理可使用 GPU，没有 CUDA 时也可运行 CPU，但速度可能较慢。

**普通用户自行拍摄的图片通常没有 GT，不能直接给出有参考的 PSNR/SSIM。** 只有同时提供对齐的参考干净图像，才适合计算这两个指标。

### 7.5 恢复或继续训练

```powershell
python -m src.train.train_dncnn
```

当前 `src/train/train_dncnn.py` 中的 `EPOCHS = 20` 是**总目标轮数**。若 `dncnn_sidd_last.pth` 已记录完成第 20 轮，则运行后会提示无需继续训练。只有主动调整总目标轮数时，才会启动新增轮次。

从 `last.pth` 恢复时，同时恢复 Adam 优化器状态和其保存的学习率；**只修改文件顶部的 `LEARNING_RATE`，不代表续训时会自动应用新学习率**。微调实验须明确设计训练起点和保存策略，避免覆盖现有最佳权重。

## 8. 当前开发进度

| 模块 / 事项 | 状态 | 备注 |
| --- | --- | --- |
| SIDD Medium 场景级数据划分 | 已完成 | 222 / 46 / 52 对，CSV 固定 |
| DnCNN 网络定义 | 已完成 | 20 层模型，可加载原权重 |
| 训练数据缓存和数据加载优化 | 已完成 | 原有缓存正常读取，结构检查通过 |
| 训练、验证、权重保存和续训 | 已完成 | 已完成 20 Epoch；断点检查通过 |
| 最佳模型 PSNR、SSIM 验证 | 已完成 | 46 对验证结果已复核 |
| 单张图像命令行去噪 | 已有实现 | 使用最佳权重，支持分块 |
| 大图推理性能验收 | 待完成 | 已有分块方案，仍需完整测速 |
| 52 对独立测试集最终评价 | 待完成 | 不应与验证集成绩混淆 |
| 其他去噪算法及公平比较 | 未实现 | 需结合开题要求确定算法 |
| Web 后端 API | 未实现 | 下一阶段拟用 FastAPI |
| 前端页面、上传和结果比较 UI | 未实现 | 后端接口明确后推进 |
| 前后端集成、部署与系统测试 | 未实现 | 后续阶段 |

## 9. 下一步开发计划

1. **建立后端最小服务**：规划并实现 FastAPI 健康检查、图片上传与输入校验，启动时加载模型。
2. **接通 DnCNN 推理 API**：复用 `src/inference.py`，让客户端上传图片并获得去噪图片和处理信息；控制文件类型、分辨率、显存占用和错误返回。
3. **测量并优化大图推理**：在 RTX 4050 6GB 上对不同分辨率测试处理时间、显存和画质，必要时调整 tile/halo/FP16 策略。
4. **开发前端页面**：完成上传、算法选择、处理状态、原图/结果图对比和文件保存等交互。
5. **扩展去噪算法**：根据开题报告确定传统方法和其他深度学习模型，统一输入输出与评价标准，实现算法间比较。
6. **正式评价与系统验收**：固定算法版本后，在保留的 52 对测试样本上计算独立指标，并进行端到端测试、毕业论文实验汇总和部署。

以上是工作计划，**不是已经完成的功能清单**。

## 10. 开发约定与注意事项

- 从项目根目录运行 `python -m ...`，减少导入路径问题；Notebook 可以从根目录或 `notebooks/` 启动。
- `src/` 放可复用的正式业务/算法代码；`scripts/` 放可重复执行的工具；`notebooks/` 放需要查看图像或指标的实验；`docs/` 放技术记录。
- 临时测试以 `test_` 开头，长期保留的检测 Notebook 以 `_test.ipynb` 结尾；不为了形式化拆分而增加无必要文件。
- 不随意修改 SIDD 划分 CSV 顺序、缓存索引、模型结构和 `.pth` 格式；变更前先备份并重新验证。
- 大型图像数据、缓存、权重文件通常应通过 `.gitignore` 排除或单独管理；**以本地现有忽略规则为准**，提交前自行检查 `git status`。
- 将**验证集**用于调参、模型选择；**测试集**留给模型确定后的最终独立评估；避免重复看测试成绩再反向修改模型。
- 当前 README 的“已完成”以已有源码及项目内实际检查为基础。后续新增后端、算法、测试结果后，应同步更新本文件。
