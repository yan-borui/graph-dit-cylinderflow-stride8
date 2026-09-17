# 32 张 L20：GLaDiT 模型规模实验

本分支 `feature/scaling-32gpu` 固定每次训练 32 张 L20，经典 DDP，每卡完整模型、
一个窗口，global batch 32。比较四个预设结构，每档三个独立训练 seed；不搜索卡数、
batch、LR 或 width/depth 矩阵。12 个任务全部并行占 384 卡，也可由调度器分批运行。

## 模型和共同配方

| Slurm array index | width / blocks | DiT 参数量 | 训练 seed |
| --- | --- | ---: | --- |
| 0–2 | 512 / 24 | 115,013,124 | 0 / 1 / 2 |
| 3–5 | 768 / 24 | 258,437,380 | 0 / 1 / 2 |
| 6–8 | 1024 / 24 | 459,140,100 | 0 / 1 / 2 |
| 9–11 | 1024 / 32 | 610,257,924 | 0 / 1 / 2 |

全部固定 H1 邻域、8 heads、MLP ratio 4、latent 4、condition 512。一帧条件联合
生成未来 64 帧，stride 8、物理跨度 5.12 秒。冻结同一份 epoch1180 UVP VGAE：
`representation_id=d0fee50b-8a47-4652-b229-0e82e06ba2d7`。

- 每次 125,000 optimizer updates，即 4,000,000 个窗口曝光。当前 B1/1M 的已有结果
  是历史参照；115M 在本轮共同配方下重训。
- AdamW：LR 1e-4、weight decay 1e-6、clip 1；4,000 updates warmup，cosine 到 1e-7。
- FP32，AMP/TF32 关闭；EMA 0.999/0.9999 都按 optimizer update 更新。
- 四档统一逐 DiT block 激活重计算，`use_reentrant=False`；DDP 使用
  `gradient_as_bucket_view=True`。每张卡仍保留完整参数、Adam 和 EMA。
- 每 5,000 updates 做 Validation24 × 3 采样 seed × raw/两套 EMA，并保存完整 checkpoint。
  每 1,000 updates 保存恢复点，每 100 updates 记日志；固定预算，无平台期早停。

每次运行的有效配置、代码副本、数据/表示身份、rank 设备信息、损失、梯度、参数量、
峰值显存、训练/验证/保存耗时及退出状态都会保留。训练 seed 和生成采样 seed 分别记录。
610M 的参数/梯度/Adam/双 EMA 状态约 13.64 GiB；激活重计算后的总占用尚待 L20 实测，
不能用参数状态或其他 GPU 的结果代替容量验收。

## 分支来源与当前验证状态

分支从 `e139e24` 创建，并带入当前 CylinderFlow 重训路径的 UVP-c4 表示支持、精确
H1 邻接 attention 和 FP32/EMA 实现。当前分支的改动不修改原 checkout 或已有训练源。
新的 scaling 代码、配置和 launcher 尚未在 32 张 L20 上运行。本轮只做静态验证，
不使用 CPU、单卡或其他 GPU 代替正式 32 卡环境的运行验收。

2026-09-17 静态检查：11 份 Python 源码通过 Python 3.10 语法解析，14 份 JSON 解析
通过；12 份 scaling 配置的模型、seed、参数量、共同配方和预算一致性检查通过。
Ruff F/E9、三个 Bash 入口分别的语法检查、diff 空白检查通过。没有执行项目入口、
GPU 测试、作业提交或训练。本地静态解析工具没有导入项目代码。

`configs/scaling_32gpu/plan.json` 固定 12 个任务的 index、配置文件和参数量。
`validate_scaling` 校验共同科学设置；正式训练入口要求完整 125k 终点。
旧版单卡/四卡配置和历史验收结果继续保留，不能作为本分支目标环境的通过记录。

## 数据与环境

在目标集群使用匹配的 CUDA/PyTorch 环境，当前源代码参考运行时为 PyTorch 2.10。
依赖见 `pyproject.toml` 与 `ENVIRONMENT.md`。保持全部节点相同环境和同一份源码；
launcher 会保存每个节点的包列表、`nvidia-smi`、GPU 拓扑和实际分配。

使用当前 epoch1180 导出的 `dit_autoencoder.pt` 与已准备的 UVP-c4 Train cache。
全部节点应能通过相同绝对路径访问代码、数据、同一份 artifacts 和结果目录；不同节点
不能分别生成 artifact_id 不同的缓存。公开旧版 `prepare.py` 默认 AE 不是本轮指定表示。
若需要在集群重新准备，使用现有表示入口并显式绑定本轮配置：

```bash
# 在已分配准备设备的环境执行一次；只读取 Train/Validation。
python -m graph_dit.representation prepare \
  --data-dir /shared/cylinderflow/stride8 \
  --autoencoder /shared/cylinderflow/dit_autoencoder.pt \
  --output-dir /shared/cylinderflow/artifacts_uvp_c4 \
  --config configs/scaling_32gpu/w512_d24_seed0.json --device cuda:0
```

沿用已经准备好的 cache 时，不重复生成。数据为 Train 1000 / Validation 100，训练取
每条轨迹的固定前 65 帧。Test 保持封存。

完整 checkpoint 同时保存 FP32 参数、两套 EMA 和 Adam 状态。25 个固定保存点、
四档各三个 seed 的主要张量合计约 2.6 TB，此外还有恢复点、预测 NPZ 和日志；结果盘需
为这些正式证据预留空间。没有自动清理 checkpoint 或失败目录。

## Slurm 启动

在本分支根目录提交。脚本默认 **4 节点 × 8 卡**，partition/account/时限由集群的
`sbatch` 参数指定，脚本不猜测这些配置。`PYTHON`、数据、artifacts、cohort 必须用所有
节点可见的绝对路径。下面路径是示例，替换为目标集群的实际挂载路径。

```bash
PYTHON=/shared/envs/gladit/bin/python
DATA=/shared/cylinderflow/stride8
ARTIFACTS=/shared/cylinderflow/artifacts_uvp_c4
COHORT=/shared/experiments/gladit_scaling32_v1

# 复用唯一的官方 preflight，所有 12 个正式结构/seed 都保持 32 张 L20。
# 缩减到 8 次更新，最大 Train 图；4 次后保存并重新加载，继续到 8 次。
# 两个端点都走原始完整 Validation24 物理评价、raw/EMA、保存和退出路径。
sbatch scripts/slurm_scaling_32gpu.sh acceptance "$PYTHON" "$DATA" "$ARTIFACTS" "$COHORT"

# 在该正式环境的验收完成并检查记录后，提交训练；不会由验收自动派生训练。
sbatch scripts/slurm_scaling_32gpu.sh train "$PYTHON" "$DATA" "$ARTIFACTS" "$COHORT"

# 例如只恢复 index 9，沿用该任务的模型、Adam、EMA、RNG 和数据游标。
sbatch --array=9 scripts/slurm_scaling_32gpu.sh resume "$PYTHON" "$DATA" "$ARTIFACTS" "$COHORT"

# 全部训练结束后，在相同 32 卡拓扑评价各自固定终点的 Validation100。
sbatch scripts/slurm_scaling_32gpu.sh evaluate "$PYTHON" "$DATA" "$ARTIFACTS" "$COHORT"
```

若实际节点是每台 4 卡，可用以下分配；总卡数、batch 和训练配方均不改变：

```bash
SCALING_GPUS_PER_NODE=4 sbatch --nodes=8 --gpus-per-node=4 \
  scripts/slurm_scaling_32gpu.sh train "$PYTHON" "$DATA" "$ARTIFACTS" "$COHORT"
```

验收和正式训练使用相同拓扑；改变节点分配后，先在新的正式环境重新使用同一个验收入口。
Slurm job array 的每个元素都是一个独立的 32 卡作业。若希望限制同时运行数量，可传
`--array=0-11%4` 等调度参数；它只影响同时调度多少次训练，不影响每次训练的 32 卡。
没有提交作业、SSH 连接、自动抢占 GPU 或停止既有实验的代码。

## 非 Slurm 集群

在分配好的每个节点各运行一次以下命令，使用相同 `MASTER_ADDR/PORT`、任务 index
和共享路径，并为各节点设置不同 `NODE_RANK=0..NNODES-1`。例如 4 节点 × 8 卡：

```bash
export NNODES=4 GPUS_PER_NODE=8
export NODE_RANK=0                  # 其他节点分别设置为 1、2、3
export MASTER_ADDR=allocated-node0  # 所有节点均可访问的 rank-zero 地址
export MASTER_PORT=29500            # 同一个独立作业共用一个空闲端口
bash scripts/scaling_32gpu_node.sh train \
  /shared/envs/gladit/bin/python 0 \
  /shared/cylinderflow/stride8 /shared/cylinderflow/artifacts_uvp_c4 \
  /shared/experiments/gladit_scaling32_v1
```

`train` 可换为 `acceptance`、`resume`、`evaluate`。每节点一个 torchrun agent，
各生成 `GPUS_PER_NODE` 个 rank；没有 elastic 自动重启。新训练拒绝复用已有 run 目录；
恢复显式使用 `resume`。验收失败保留证据，下一次验收使用新 cohort 目录。
恢复所需源码与配置必须和该 run 冻结副本一致。相同运行用文件锁防止重复写入。

## 评价与三张规模曲线

`selection_endpoint.json` 只在 125k 端点的 raw/EMA 三个候选之间使用现有 Validation24
严格规则选权重。`selection.json` 保留全程最佳，另列实际更新数。主规模表采用固定终点，
不以各自全程最优时刻混合比较。`evaluate` 使用前者，对 Validation100 分片到 32 卡；
生成 `validation/<task>/endpoint/summary.json`、逐轨迹指标及预测，并可复用已完成的评价分片。
需要报告全程最佳时，可在同样 torchrun 分配下调用 `graph_dit.scaling_evaluate --selection best`，
它使用单独的 `best/` 结果目录。

```bash
python -m graph_dit.scaling_report --cohort-dir "$COHORT" \
  --output-dir /shared/reports/gladit_scaling32_v1
```

报告生成：

- `runs.csv`：12 次训练全部列出，含失败/待运行、固定终点权重、Validation24/100 和卡时。
- `endpoints.csv`、`quality_vs_parameters.png/.pdf`：共同 125k 终点的参数量—质量；
  独立训练 seed 的均值/样本标准差和已完成 seed 数。缺少任意种子时明确标记未完成。
- `curves_validation24.csv`、`quality_vs_updates.png/.pdf`：相同数据曝光下的物理质量曲线。
- `quality_vs_gpu_hours.png/.pdf`：复用相同检查点，按实测训练 GPU-hours 展示成本。
  不把 GPU-hours 标为 FLOPs；精确 FLOPs 需另外统计当前稀疏 H1 和重计算实际运算。

主指标为 `cylinderflow.physical_mesh.v1` 面积加权未来 64 帧 UV relative RMSE，
每轨迹对三个采样 seed 平均，再对轨迹等权平均。完整评价文件同时保留压力、涡量、散度、
能谱、相位等物理分项、失败和不利轨迹。Validation100 是完整验证集汇总，不标作新 Test。
结论限定为该预设模型族、数据和共同配方；四个规模无需另行调出各自最优超参数。

方法依据：[DiT](https://arxiv.org/html/2212.09748v2) 的共同训练配方与固定 token 规模比较，
以及 [PDE-Transformer](https://arxiv.org/html/2505.24717v1) 的固定其余架构加宽实验。

## 记录位置

| 内容 | cohort 内路径 |
| --- | --- |
| 各节点启动日志、包版本、拓扑、退出码 | `launcher/<task>/<action>/node_<rank>/attempt_*/` |
| 8 步验收及恢复尝试 | `acceptance/<task>/attempt_001/`、`attempt_002/` |
| 正式运行、源快照、完整配置、rank 设备表 | `runs/<task>/` |
| 每步/定期训练日志和异常 | `runs/<task>/attempt_*/` |
| Validation24 逐权重记录 | `runs/<task>/evaluation_records/`、`monitor/` |
| 固定终点的 Validation100 | `validation/<task>/endpoint/` |

缩减验收只建立它实际覆盖的启动、加载、梯度、EMA、保存/恢复和评价证据，不能保证
125k 更新全程稳定或科学效果。正式运行保留所有失败与不利结果。
