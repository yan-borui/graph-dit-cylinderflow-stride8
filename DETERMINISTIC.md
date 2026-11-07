# 确定性图 Transformer 训练

本分支回答在相同图表示和时空 Transformer 上，扩散训练与迭代生成相对一次直接回归带来什么变化。两种模型使用同一份骨干实现：24 层、宽度 512、8 个头、MLP 比例 4，动力学模型有 115,013,124 个可训练参数。在同一粗图上，每个节点都可以关注自身与一跳邻居在所有 65 个时间位置的 token。

## 运行环境与资产

使用已有的 Linux CUDA 环境（含 PyTorch、PyG、NumPy、SciPy、h5py 和本工程声明的依赖）。从仓库根目录运行脚本，保留调度器分配的 `CUDA_VISIBLE_DEVICES`。默认训练为单机四进程 NCCL。输入模型为对应数据集的投稿 VGAE，表示与物理、latent 两套归一化必须来自同一份导出。

| 环境变量 | 含义 |
| --- | --- |
| `PAPER_ARTIFACTS` | 原论文 representation artifacts 目录，内含 `artifact.json`（含 latent_mean/std、物理归一化、表示和资产 ID）及 `autoencoder.pt` 或 `dit_autoencoder.pt`。准备与训练时必填。 |
| `PAPER_REPRESENTATION_ID`、`PAPER_ARTIFACT_ID` | Airfoil 必填，从论文选定的 checkpoint 或评价记录独立取得同一组 `representation_id`、`artifact_id`。前者绑定冻结模型，后者绑定该次缓存的 latent 统计。CylinderFlow 已固定两个 ID，对应投稿的 epoch1180 表示。 |
| `WORK_DIR` | 默认 `runs/deterministic_joint`，按 `airfoil`、`cylinderflow` 建立独立目录。可改到共享存储。 |
| `DATA_DIR`、`CACHE_DIR`、`RESULT_ROOT` | 可分别指定 Train/Validation 数据根、确定性 latent 缓存、单次训练结果根。身份匹配的 Train65 缓存可复用。Transformer 默认结果目录为 `$WORK_DIR/<dataset>/transformer_seed0`，resume 指回同一目录。 |
| `RAW_DIR`、`OFFLINE` | 官方 `meta.json`、`train.tfrecord`、`valid.tfrecord` 所在目录。未提供时在数据根下准备官方数据。设置 `OFFLINE=1` 时只使用已有文件。 |
| `TEST_DATA_DIR`、`SPLIT`、`OUTPUT_DIR` | Test 可使用独立数据根。评价分支为 `validation` 或 `test`；每次运行写入新的输出目录。 |

## 训练与恢复

```bash
export WORK_DIR=/shared/runs/deterministic_joint
# 例：在 Airfoil 的四卡分配中
export PAPER_ARTIFACTS=/shared/paper/airfoil/artifacts
export PAPER_REPRESENTATION_ID="$AIRFOIL_PAPER_REP_ID"
export PAPER_ARTIFACT_ID="$AIRFOIL_PAPER_ARTIFACT_ID"
bash scripts/deterministic_4gpu.sh airfoil train
# 从相同 RESULT_ROOT 的完整恢复点继续
bash scripts/deterministic_4gpu.sh airfoil resume
```

CylinderFlow 使用自己论文版权重目录，数据集参数改为 `cylinderflow`。训练自动执行 `prepare`；也可单独运行 `bash scripts/deterministic_4gpu.sh airfoil prepare`。准备阶段使用一张可见 GPU 编码 Train，检查论文表示与统计，在锁内发布完整缓存。两个并发任务应分到各自的卡。输入不匹配会报告缺少或不匹配的资产，应换用正确导出或新的缓存目录。

训练按每卡一条轨迹、累积 1 次、全局 batch 4，使用 FP32 AdamW，学习率从 1,000 步 warmup 后的 `1e-4` 余弦衰减至 `1e-7`，权重衰减 `1e-6`、梯度裁剪 1。25 万更新共看 100 万窗口。每条轨迹上的 MSE 对全部 64 帧、latent 节点和通道取平均，四个 rank 对四条轨迹等权。两套 EMA 的 0.999 和 0.9999 按窗口换算到每次更新。每 1,250 步保存恢复点，每 12,500 步在固定 Validation24 用完整解码物理速度相对误差选择 raw/EMA。保留每个候选的逐轨迹结果、失败数、最优权重路径、参数量、rank 设备和训练记录。

训练从新初始化的 Transformer 开始。Airfoil 沿用论文四卡方案；CylinderFlow 投稿主结果使用单卡、全局 batch 1。比较该结果时应分别报告 batch、更新次数和窗口曝光。与相同四卡配方训练的扩散模型比较，可以进一步控制优化设置。

## 数据与预测

每条轨迹只取第 0 帧和未来第 1–64 帧（raw 0, 8, …, 512）。新缓存只存这 65 个 posterior means。历史 VGAE 及其绑定归一化可以来自 Train 的 75 帧，这项历史暴露记录在缓存元数据中。官方 Train 1000 条、Validation 100 条。单独执行 Test 时使用其独立的 100 条轨迹。

输入序列的第 0 个位置填实际首帧的标准化 latent，第 1–64 个位置填零。每个位置仍有原模型的粗节点条件、空间坐标和物理时间位置编码。原 AdaLN 调制网络在所有位置接收相同的固定基准嵌入，保留其可训练参数。一次前向直接输出 `[64, N_latent, 4]`，通过 MSE 拟合标准化的真实未来 latent。训练与推理都走这一条前向路径；模型没有噪声日程、随机加噪或去噪采样循环。

解码时逐帧撤销 latent 归一化，再还原物理 UVP。预测 API 只接收首帧、节点条件、位置和图拓扑；评估在生成之后读取真值。Airfoil（dt 0.0016）预测所有节点；CylinderFlow（dt 0.08）将入口和壁面速度回填为首帧已知值，压力及出口仍取模型输出。

## 评价与成本

### 下载 Test 原始数据

从仓库根目录运行，使用 Python 3.10+ 标准库即可，无需 GPU、TensorFlow 或模型权重：

```bash
bash scripts/download_test.sh --dataset both --raw-root /shared/data/mgn-test-raw
```

也可直接运行 `python scripts/download_test.py --dataset both --raw-root /shared/data/mgn-test-raw`。只下载一个数据集时将 `both` 改为 `airfoil` 或 `cylinderflow`。脚本只获取官方 `meta.json` 和 `test.tfrecord`，分别保存到根目录下的 `airfoil/` 与 `cylinder_flow/`，两个 Test 文件合计约 6.41 GB。已有文件复用；中断下载保留 `.download` 文件，重跑同一命令继续传输。目录锁与评测转换入口共用。

原始文件可供各 baseline 共用，各方法仍使用自己的转换器和训练时确定的归一化。该命令仅下载；本分支在首次 Test 评价时自动生成 `TEST_DATA_DIR/test65/` 下的 HDF5 和 manifest，保留 raw 0, 8, …, 512。转换沿用所选 VGAE 绑定的 Train 统计。

### 全量评分与测速

```bash
# 沿用以上 WORK_DIR 与数据集身份设置；示例均使用 Airfoil。
bash scripts/deterministic_4gpu.sh airfoil eval
RAW_DIR=/shared/data/mgn-test-raw/airfoil \
  SPLIT=test TEST_DATA_DIR=/shared/data/airfoil-test OFFLINE=1 \
  bash scripts/deterministic_4gpu.sh airfoil eval
bash scripts/deterministic_4gpu.sh airfoil benchmark
```

CylinderFlow 的 Test 命令使用 `RAW_DIR=/shared/data/mgn-test-raw/cylinder_flow`、`TEST_DATA_DIR=/shared/data/cylinderflow-test`，数据集参数改为 `cylinderflow`，并沿用它自己的 `WORK_DIR`、`CACHE_DIR` 与 `RESULT_ROOT`。Airfoil 继续使用同一投稿导出的两个身份 ID。`OFFLINE=1` 使上述评分只读取已经下载的原始文件；下载阶段与训练阶段分别执行。

Test 使用 Validation 选定的权重，完整评价官方 100 条轨迹。Test 与原 Validation 结果分别保留并标明 split；补测结果不覆盖已报告的 Validation 数值。跨方法比较时对齐 split、帧范围、物理量、采样预算和指标定义。

两项命令都从 `selection.json` 加载 Validation 选定的 checkpoint 和 raw/EMA 类型。评价输出 100 份逐轨迹记录和 `summary.json`，覆盖全窗速度相对误差、去均值压力 RMSE、涡量、散度、动能、enstrophy 和逐帧指标。评分排除首帧，各轨迹等权。有失败时保存全体轨迹及失败数，各指标同时标明有限值数量，该次评价标为不完整并返回非零退出码。

测速使用单张可见 GPU、固定 Validation24，每条轨迹预热两次、计时三次。计时从 CPU 首帧和预加载的静态网格开始，覆盖传输、编码、一次图 Transformer 前向、64 帧解码、边界处理和 CPU 返回。报告含端到端延迟、峰值已分配显存及设备信息。与 GLaDiT、MGN、AROMA 比较时使用相同硬件与精度。

本分支发布时完成静态检查。正式四卡 NCCL、峰值训练显存及运行时吞吐须在目标机器核实。在已核实的等价四卡分配中，可设置 `VERIFICATION_UPDATES`（例如 2）并指定独立的 `RESULT_ROOT`，用同一入口完成一次缩量训练、验证和保存；使用相同设置执行 `resume` 可检查恢复。缩量结果不作为实验指标。
