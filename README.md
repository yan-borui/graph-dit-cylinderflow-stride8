**Test 评测：**[官方下载、1+64 转换与完整 Test100 入口](TEST.md)。

# 确定性 GNN U-Net 的四卡联合轨迹预测

冻结各数据集论文使用的 VGAE，从首帧与静态网格一次回归后续 64 帧的绝对 latent。模型沿用三级多尺度 GNN U-Net（宽 128，各层消息传递深度 1、2、2），每个粗节点输出 64 × 4 个值，再由原解码器预测物理 UVP。该对照同时改变了动力学骨干和训练目标。

在具备四张 NVIDIA GPU、NCCL 和原项目依赖的 Linux 环境中，从仓库根目录运行：

```bash
# 两个目录分别来自对应数据集原论文模型的 representation artifacts。
PAPER_ARTIFACTS=/shared/paper/cylinderflow/artifacts \
  bash scripts/deterministic_4gpu.sh cylinderflow train
PAPER_ARTIFACTS=/shared/paper/airfoil/artifacts PAPER_REPRESENTATION_ID="$AIRFOIL_PAPER_REP_ID" \
  PAPER_ARTIFACT_ID="$AIRFOIL_PAPER_ARTIFACT_ID" \
  bash scripts/deterministic_4gpu.sh airfoil train
```

先将示例路径换成实际资产位置。Airfoil 的两个 ID 分别取自投稿模型选中 checkpoint 的 `representation_id`、`artifact_id`（或其独立评价记录）。仓库不分发这些权重。两个任务分别使用分配的四张卡，结果与缓存按数据集隔离。准备、恢复、Test、测速及数据契约见 [确定性模型使用说明](DETERMINISTIC.md)。

## 原 Airfoil DiT 入口

**NAS 运行：**[启动入口与共享目录设置](NAS.md)。

**多次采样：**[Validation100 均值、方差与成本](SAMPLING.md)。

`feature/airfoil-uvp-4gpu`提供默认四卡的Airfoil训练与恢复入口。
完整数据契约、环境、固定配方和命令见[Airfoil操作说明](AIRFOIL.md)。

GLaDiT使用已锁定的5090模型配方；本分支只训练一个seed0配置。

设置原始数据目录和四卡训练结果目录后，一条命令自动完成缺文件下载、
全量stride8转换、Train归一化、模型缓存准备及训练：

```bash
export RAW_DATA_DIR=/data/datasets/meshgraphnets/airfoil
export RESULT_ROOT=/shared/runs/airfoil_dit_seed0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用Slurm时保留调度器分配的值
bash scripts/airfoil_4gpu.sh train
```

已有数据/缓存自动复用，`resume`恢复训练。只处理全量数据时运行
`bash scripts/airfoil_4gpu.sh data`，无需GPU。DiT另需设置`AUTOENCODER`为已训练的Airfoil VGAE导出。

此分支已做静态检查，目标四卡运行验收尚未执行。Test 入口见 [TEST.md](TEST.md)。
原上游源码、许可证和历史CylinderFlow记录保留。
