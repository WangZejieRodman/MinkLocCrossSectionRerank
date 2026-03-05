MinkLocCrossSection: Coarse-to-Fine Dual-Stream

单帧级粗精双流：基于横截面切片序列的地下/长廊点云位置识别

本项目基于 MinkLocCrossSection 进行了深度架构升级。针对长廊、地下巷道等缺乏显著局部几何特征的狭长环境，我们提出了一种**“单帧级粗精双流（Coarse-to-Fine Dual-Stream）”**网络架构。

该架构将传统的“跨帧时序”降维打击为“单帧内的空间序列”，在完全不增加建图/推理时延（保持单帧输入）的前提下，利用动态时间规整（DTW）解决了纵向位移偏差问题，在 CYD 巷道数据集上实现了 Recall@1 超 10% 的绝对性能飞跃。
🌟 核心创新与优势 (Highlights)

    解决信息混叠 (Channel Mixing): 在原始框架中，沿行驶方向（S轴）的 64 个切片序列在 2D 稀疏卷积中被压缩进通道维度并彻底“搅浑”。我们开辟了平行的轻量级提取分支，通过“批次编码隔离法”完美保留了 64 个有序切片的空间几何先验。

    单帧提取“伪时序” (Intra-frame Spatial Sequence): 将一帧单帧点云解析为形状为 (64, N) 的独立特征序列，天然契合序列比对算法，完全摒弃了传统 SeqSLAM 复杂的跨帧拼接逻辑。

    逆序 DTW 对齐增强双向鲁棒性 (Bidirectional DTW): 利用可微的 Soft-DTW Triplet Loss 进行端到端训练。在推理重排（Re-ranking）阶段，通过正向与逆向的双向 DTW 距离计算，赋予了模型对“掉头/反向行驶”天生的 180° 偏航角绝对鲁棒性。

    粗精结合的极致性能 (Coarse-to-Fine Retrieval):

        粗流 (Coarse): 256维全局向量 + KD-Tree，极速召回 Top-25 候选帧。

        精流 (Fine): 提取 Top-25 的 64维切片序列，利用 DTW 算法吸收纵向漂移误差，进行精准重排。

🚀 性能基准 (Benchmarks)

我们在极具挑战性的地下巷道数据集 CYD Dataset 上进行了评估。引入精流重排后，召回率获得了惊人的提升：
检索阶段	Recall@1	Recall@5	Recall@10
阶段一：粗流 Baseline (KD-Tree)	71.52%	86.67%	93.94%
阶段二：精流 DTW 重排 (Dual-Stream)	

81.82%

(+10.30%)
	

95.15%

(+8.48%)
	

96.97%

(+3.03%)

    📌 结论: 粗精双流架构在极小的额外计算开销下，将 Top-1 命中率提升了 10.3%，Top-5 命中率逼近 95% 的理论极限。

🛠️ 环境依赖 (Dependencies)

    Ubuntu 18.04/20.04

    Python 3.8+

    PyTorch 1.8+

    MinkowskiEngine 0.5.4

    scikit-learn, tqdm, numpy

📂 核心代码架构修改指南

如果你是从旧版本迁移，本项目主要在以下核心文件进行了重构：

    models/slice_branch.py (新增): 轻量级 2D 稀疏卷积切片序列提取器。

    models/losses/soft_dtw.py (新增): 支持 PyTorch Autograd 的防 In-place 报错版 Batch Soft-DTW。

    models/minkloc.py & models/model_factory.py: 模型顶层结构解耦，支持输出 {'global': ..., 'sequence': ...} 双特征字典。

    models/losses/loss.py: 结合 TruncatedSmoothAP 与 Soft-DTW 的双流联合损失 DualStreamLoss。

    training/trainer.py: 修复了支持 Multistage（分段梯度累加）的双路梯度反向传播逻辑。

    eval/evaluate_cyd.py: 重构了评测逻辑，集成了极速 Numpy 版 DTW 与空切片过滤（Empty Slice Masking）机制，实现两阶段漏斗式检索。

🏃 快速开始 (Quick Start)
1. 准备数据集 (CYD Dataset)

请在 config/config_cyd_cross.txt 中配置您的数据集路径：
Ini, TOML

dataset_folder=/path/to/your/CYD/
train_file=datasets/cyd/training_queries_cyd.pickle
val_file=datasets/cyd/test_queries_cyd.pickle

2. 训练模型 (Training)

运行以下命令启动端到端的双流联合训练。系统会自动提取特征并计算 Loss = L_coarse + L_fine。
Bash

export OMP_NUM_THREADS=12
python training/train_cyd_cross.py

3. 评估与重排 (Evaluation & DTW Re-ranking)

在 eval/evaluate_cyd.py 底部修改为你的最新模型权重路径，然后执行：
Bash

python eval/evaluate_cyd.py

评测脚本将依次输出“阶段一 (KD-Tree Baseline)”和“阶段二 (DTW 逆序双向增强)”的性能对比指标。
📝 待办事项 (To-Do)

    [x] 完成单帧 64 切片的通道隔离 (Slice Sequence Branch)

    [x] 实现纯 PyTorch 版可微 Batch Soft-DTW

    [x] 解决多批次（Multistage）训练下的梯度流断裂问题

    [x] 引入基于 DTW 的两阶段评估逻辑

    [ ] 开展序列特征维度 N 对显存及 Recall 性能的消融实验

    [ ] 输出 Query 与 Database 切片序列的 2D 匹配路径热力图 (Warping Path Visualization)

🎓 引用 (Citation)

如果您在研究中使用了本项目的代码或思路，请考虑引用原始仓库以及本项目的创新思路。

原项目: WangZejieRodman/MinkLocCrossSection
