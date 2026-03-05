# MinkLocCrossSection

MinkLocCrossSection 是一个专为狭长环境（如地下巷道、走廊）设计的点云位置识别（Place Recognition）框架。该项目针对传统鸟瞰图（BEV）投影在长廊环境中存在的局限性，提出了一种创新的**横截面量化（Cross-Section Quantization）**方法。通过结合环境的几何拓扑（中心线），将三维点云“拉直”并投影至 (W, Z, S) 坐标系，再利用 MinkowskiEngine 进行高效的稀疏卷积处理。

🔥 **最新重大更新**：本项目现已全面升级为**单帧级粗精双流（Coarse-to-Fine Dual-Stream）**架构。通过在单帧点云内部挖掘“切片空间序列”，并引入 Soft-DTW（软动态时间规整）进行端到端优化与重排，在 CYD 巷道数据集上实现了 Recall@1 超过 **10% 的绝对性能飞跃（71.5% ➔ 81.8%）**！

本项目在 CYD（巷道）数据集上进行了训练与验证。

## 🌟 主要特性

* **横截面量化 (Cross-Section Quantization)**: 提取点云的中心线主分支，计算累积弧长与切线方向，将点云沿着中心线展开并投影为 S（沿着中心线）、W（横向偏移）、Z（高度）坐标。这种表示方式极大地保留了巷道的截面特征。
* **单帧级粗精双流 (Coarse-to-Fine Dual-Stream) 🚀**:
* **粗流 (Coarse Stream)**：将信息压缩为 256 维全局特征，利用 KD-Tree 极速召回 Top-25 候选帧。
* **精流 (Fine Stream)**：通过巧妙的批次编码隔离机制，平行提取 S 轴上 64 个切片的独立特征，构成 `(64, N)` 的有序序列。对 Top-25 候选进行极速 DTW 序列比对与重排，**无需跨帧累积即可享受序列匹配的超强鲁棒性**。


* **双向行驶鲁棒性 (Bidirectional Robustness)**: 针对巷道中车辆可能正向或反向行驶导致的序列完全倒置问题，在精流评估中无缝集成了“逆序 DTW 匹配”逻辑，赋予模型对行驶方向和 180° 偏航角的绝对鲁棒性。
* **高效稀疏卷积**: 基于 `MinkowskiEngine` 构建的 `MinkBEVBackbone` (2D 稀疏卷积网络)，在大幅降低显存占用的同时保持了高分辨率的空间特征。
* **双路联合度量学习优化**: 粗流采用 `TruncatedSmoothAP` 损失直接优化 Recall@k；精流引入 `Soft-DTW Triplet Loss` 拉近正样本切片序列并推远负样本，实现了完美分离的端到端联合训练。

## 📁 目录结构

```text
MinkLocCrossSection/
├── config/                  # 配置文件目录
│   └── config_cyd_cross.txt # CYD横截面模型训练配置文件
├── datasets/                # 数据集处理与加载
│   ├── cyd/                 # CYD数据集专用的正负样本字典生成脚本
│   ├── augmentation.py      # 数据增强（翻转、随机丢弃等）
│   ├── base_datasets.py     # 基础数据加载器定义
│   └── cross_section_quantization.py # 核心：横截面量化器实现
├── eval/                    # 评估脚本
│   ├── evaluate_cyd.py      # 双流评估（KD-Tree粗召回 + DTW精流重排）
│   └── evaluate_cyd_rotation.py # 旋转鲁棒性评估（测试0°~180°旋转干扰）
├── models/                  # 模型定义
│   ├── minkloc.py           # 双流顶层网络结构封装
│   ├── minkbev.py           # 基于Minkowski的粗流稀疏骨干网络
│   ├── slice_branch.py      # ✨ 新增：轻量级切片序列特征提取分支
│   ├── layers/              # 池化层（GeM, NetVLAD）与注意力模块
│   └── losses/              
│       ├── loss.py          # 双流联合损失函数 (DualStreamLoss Wrapper)
│       ├── soft_dtw.py      # ✨ 新增：支持梯度回传的批处理 Soft-DTW
│       └── truncated_smoothap.py # SmoothAP 损失
├── training/                # 训练逻辑
│   ├── train_cyd_cross.py   # 训练入口文件
│   └── trainer.py           # 支持多阶段梯度缓存(Gradient Accumulation)的双流训练循环
└── misc/                    # 工具函数 (参数解析等)

```

## 🛠️ 环境依赖

推荐使用 Anaconda 构建环境：

* Python 3.8+
* PyTorch (需与你的 CUDA 版本匹配)
* **MinkowskiEngine** (核心依赖，用于稀疏卷积)
* pytorch-metric-learning (用于损失函数)
* scikit-learn (用于 KDTree 生成正负样本)
* pandas, numpy, tqdm, scipy

## 🚀 快速开始

### 1. 数据准备 (CYD 数据集)

确保 CYD 数据集按照以下结构组织：

```text
/home/wzj/pan2/巷道/CYD/cyd_NoRot_NoScale/
  ├── 100/
  │   ├── pointcloud_20m_10overlap/ (存放 .bin 点云)
  │   ├── centerline/ (存放对应的 _centerline.txt 中心线文件)
  │   └── pointcloud_locations_20m_10overlap.csv
  ├── ...

```

生成训练和测试用的 Pickle 字典：

```bash
# 生成训练字典 (Session 100-108)
python datasets/cyd/generate_training_tuples_cyd.py

# 生成测试字典 (Database: 109-111, Query: 112-113)
python datasets/cyd/generate_test_sets_cyd.py

```

### 2. 训练模型

编辑 `config/config_cyd_cross.txt`，检查 `dataset_folder`、`train_file` 和 `val_file` 等路径是否正确。

运行训练脚本：

```bash
python training/train_cyd_cross.py

```

训练过程中的日志、显存使用量、体素统计以及双流损失（Coarse & Fine）的变化会被记录在本地的 `trainer.log` 文件中。

### 3. 模型评估

**标准评估（粗精双流检索）：**
编辑 `eval/evaluate_cyd.py`，将 `weights_file` 指向你训练好的模型权重路径，然后执行：

```bash
python eval/evaluate_cyd.py

```

程序将输出两个阶段的评估结果，你可以直观地看到精流重排带来的巨大提升：

1. **第一阶段：粗流 (KD-Tree Baseline)**：仅使用全局特征的传统 Recall 指标。
2. **第二阶段：精流 (DTW Re-ranking)**：经过切片序列 DTW 对齐及逆序双向增强后的最终 Recall 指标。

**旋转鲁棒性测试：**
为了验证模型对偏航角变化的鲁棒性，运行：

```bash
python eval/evaluate_cyd_rotation.py

```

该脚本会将查询点云及中心线在 Z 轴上旋转特定角度（如 5°, 15°, 90°, 180°）后与未旋转的数据库进行匹配，以验证横截面投影的视角不变性。
