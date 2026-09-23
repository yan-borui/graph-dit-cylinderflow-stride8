# 新训练种子

在已有 CylinderFlow 正式运行保存的冻结源码上，从头训练一个新 DiT 种子。
本目录提供该源码的启动补充文件。复用同一 VGAE、Train 缓存、生产环境、
单张 RTX 5090、FP32、B1、H1/512×24、1M 更新预算及原 Validation 选优规则。
配置仅将 DiT 训练种子改为 1，冻结 VGAE 的种子保持 0。

## 准备独立运行

将原正式运行的冻结源码复制到独立源码目录，并保留其原始配置及
`cylinderflow_upstream.json`。将本目录的 seed1 配置复制到该源码的 `configs/`，
三个 shell 入口复制到 `scripts/`。运行使用这份独立源码中的训练、表示和验收模块。

NAS 使用本交付的 [NAS 说明](../../NAS.md)。在独立源码副本准备好后，从本交付仓库运行：

```bash
: "${REPO_DIR:?Set REPO_DIR to the independent source copy for this new seed}"
python campaigns/training_seed/adapt_nas_source.py --source "$REPO_DIR"
cp scripts/nas.sh "$REPO_DIR/scripts/nas.sh"
```

适配器备份并替换源码副本中的运行目录锁，加入标准库目录锁实现。源身份会随之改变；新训练与验收使用全新目录，原运行保留原源码。

激活原生产环境后，设置 `PYTHON`、`DATA_DIR`、`AUTOENCODER` 和 `ARTIFACTS_DIR`。
VGAE 使用原选优导出，缓存绑定同一表示和 Train 统计。为新实验分配约 70 GB 空间，
保存历史 checkpoint、完整恢复点、评价产物及验收记录。

```bash
export EXPERIMENT_DIR=/shared/experiments/cylinderflow_seed1
export REPO_DIR="$EXPERIMENT_DIR/source"
export CONFIG="$REPO_DIR/configs/h1_w512_d24_cosine1m_uvp_c4_seed1.json"
export RUN_DIR="$EXPERIMENT_DIR/run"
export ACCEPTANCE_DIR="$EXPERIMENT_DIR/acceptance"
export LAUNCH_DIR="$EXPERIMENT_DIR/launcher"
cd "$REPO_DIR"
mkdir -p "$LAUNCH_DIR"
nohup setsid bash scripts/nas.sh bash scripts/launch_training_seed.sh </dev/null >"$LAUNCH_DIR/bootstrap.log" 2>&1 &
```

入口依次执行表示验证、既有八步保存恢复验收和新种子正式训练。
验收保留原模型、精度、设备和完整预测长度，包含真实 Validation 物理评价。
每阶段保存 PID、日志和退出状态。正式训练每 50k 更新评价 Validation24，
原始权重及两套 EMA 沿用原有评分与选优。Test 保持封存。

`retrain_h1.sh` 可通过 `CONFIG` 选择种子配置，默认仍使用冻结源码原 seed0 配置。
新种子使用新的运行目录；原运行继续保留其模型、源码和恢复状态。

## 原生产运行证据

目标机器已通过八步验收，包括第 4 步保存后恢复到第 8 步，以及原始权重、
两套 EMA 的 18 段真实物理评价。新种子已从头进入正式训练。
部署核对确认 48 份生产 Python 文件与 seed0 冻结源码逐字节一致，配置仅改变训练 seed。
后续收敛结果由完整训练及原协议 Validation 提供。
