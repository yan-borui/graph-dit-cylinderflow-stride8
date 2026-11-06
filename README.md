# 确定性图 Transformer 的四卡联合轨迹预测

本分支 `feature/deterministic-graph-transformer-4gpu` 实现 GLaDiT 的去扩散消融。模型冻结投稿 VGAE，复用 GLaDiT 的 24 层、宽度 512、8 头、一跳图注意力 Transformer。从首帧和静态网格一次回归未来 64 帧的绝对 latent，再用原解码器还原物理 UVP。扩散版和回归版共享相同的 token 编码、时空位置、注意力块和输出头。

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

此分支已做静态检查，目标四卡运行验收尚未执行。Test保持封存。
原上游源码、许可证和历史CylinderFlow记录保留。
