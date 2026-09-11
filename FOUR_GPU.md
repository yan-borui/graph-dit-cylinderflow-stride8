# H1：12 组四卡窗口预算搜索（2026-09-12）

本轮用 48 张同型卡并行跑 12 组 H1，每组四卡 DDP；另外 16 张留给四个 baseline。
配置入口为 `configs/search_4gpu_12_20260911.json`。原 `configs/base.json`、72 组单卡
搜索及已完成的 4090 运行继续使用各自配置。本轮全部从 seed 0 起跑。

## 配置与计时单位

四种模型/LR 组合分别交叉三个 cosine 终点，共 12 组：

| width | blocks | heads | 峰值 LR | cosine 终点（全局窗口） |
| --- | --- | --- | --- | --- |
| 512 | 16 | 8 | 1e-4 | 250k / 500k / 1M |
| 512 | 24 | 8 | 1e-4 | 250k / 500k / 1M |
| 512 | 24 | 8 | 5e-4 | 250k / 500k / 1M |
| 512 | 24 | 8 | 1e-3 | 250k / 500k / 1M |

这里一个窗口是一个轨迹的首帧条件与未来 64 帧。每卡 microbatch=1，累积=1，
global batch=4；每次优化器更新消耗四个窗口。每组实际预算固定 **250,000 窗口 =
62,500 updates**；三个日程终点是 62,500 / 125,000 / 250,000 updates。

| 项目 | 全局窗口 | optimizer updates |
| --- | ---: | ---: |
| warmup | 4,000 | 1,000 |
| 日志 | 每 100 | 每 25 |
| 滚动恢复点 | 每 5,000 | 每 1,250 |
| 独立 checkpoint 与物理 Validation | 每 50,000 | 每 12,500 |
| 本次每组预算 | 250,000 | 62,500 |

H1 一跳注意力，DiT/VGAE FP32，关闭 autocast 与 TF32。AdamW 保持 weight decay
1e-6、clip 1.0 和原 betas/eps。warmup 后纯 cosine 的绝对最低 LR 为 **1e-6**，
所有峰值 LR 共用这个下限。按全局窗口 $n$、warmup $W=4000$、预定终点 $U$：

$$
\eta(n)=10^{-6}+\frac{\eta_{\max}-10^{-6}}{2}
\left[1+\cos\left(\pi\frac{n-W}{U-W}\right)\right],\quad W<n\le U.
$$

500k/1M 日程在本次 250k 窗口预算结束时尚未退火完成。恢复不改变日程，也不自动
晋级或延长；关闭平台期早停。最终 B1 训练另行决定，筛选 B4 的排序需在最终配方中复核。

同时维护 raw、EMA0.999、EMA0.9999。EMA 标称衰减按窗口计，四卡每次更新实际使用
`0.999**4` / `0.9999**4`。这是保持平滑窗口尺度的换算，B4 的参数轨迹仍与 B1 不同。
旧配置缺少 `ema_decay_unit` 时继续按 update 更新 EMA。

## 准备、容量检查和启动

使用 [ENVIRONMENT.md](ENVIRONMENT.md) 中已有环境；下列命令在本仓库根目录执行。
`DATA_DIR` 与 `ARTIFACTS` 指向 `python prepare.py --device cuda:0` 官方入口生成的
数据和表示目录。共享 Train75 VGAE/缓存/Train 归一化和 Train/Validation 身份校验保持不变。
DiT 仍仅使用缓存的前 65 帧，物理预测跨度 5.12 秒。Test 封存。

```bash
python -m graph_dit.campaign plan \
  --search configs/search_4gpu_12_20260911.json \
  --output-dir campaigns/h1_4gpu_12
```

先从生成的 `plan.json` 取 d16 和 d24 的实际 config 路径，各用新结果目录执行目标四卡检查。
`CONFIG` 指向其中一份配置；`GPU_IDS` 是该作业获配的四个 ID/UUID。

```bash
CUDA_VISIBLE_DEVICES="$GPU_IDS" python -m torch.distributed.run \
  --standalone --nproc-per-node=4 --max-restarts=0 --module graph_dit.preflight \
  --config "$CONFIG" --data-dir "$DATA_DIR" --artifacts "$ARTIFACTS" \
  --output-dir runs/preflight_4gpu_new --updates 8
```

检查走实际 DDP 更新路径：每卡最大 Train 图、raw/两套 EMA 参数变化、非零 attention
梯度、完整 64 帧预测，并在结束时执行一轮 216 段物理 Validation。
输出每卡显存和实际耗时；这些是容量/执行证据，不能据 8 步判断收敛。

Slurm 单节点四卡作业数组：

```bash
sbatch scripts/slurm_screen_4gpu.sh "$PYTHON" \
  "$PLAN_JSON" "$DATA_DIR" "$ARTIFACTS"
```

脚本声明 array 0–11，每个任务申请四卡。四个 baseline 分别用各自四卡脚本提交，
总并发为 64 张卡。分区、时限和账户由集群实际分配指定。
在一台能直接看到全部获配 GPU 的机器上，也可用：

```bash
python -m graph_dit.campaign run-local --plan "$PLAN_JSON" \
  --data-dir "$DATA_DIR" --artifacts "$ARTIFACTS" \
  --gpus "$H1_GPU_IDS" --resume-incomplete
```

`H1_GPU_IDS` 为 48 个互不重复、已分配的 ID/UUID；启动器按相邻四个分成一组。
本地分组入口只使用本机设备，不跨主机拼组。每个任务内由 torchrun 管理四个 rank。

## Validation、恢复与结果

每个 checkpoint 使用固定 Validation-24 × sampling labels 0/1/2 × 三种权重，
共 216 段。20-step DDIM、仅评分未来帧，UV 面积加权整段 relative RMSE 先在每条轨迹
平均三个采样，再对 24 条轨迹等权平均。四卡按轨迹分片，合并前检查完整样例集合。
每种权重的 72 段全部有效才可选优，分数严格下降更新最佳；持平保留较早更新及稳定权重顺序。
保留压力、涡量、边界、逐样例结果与完整预测。五轮/组，共 **12,960 段**。

`recovery_latest.pt` 保存 raw、EMA、Adam、每卡 Python/NumPy/Torch/CUDA RNG、
扩散 generator、全局窗口游标、不可变配置及表示身份。源码保存在 run 的 `source/`，
恢复拒绝源码、world size、配置或表示身份变化。旧单卡 checkpoint 仍由旧路径读取；
四卡训练使用 `graph_dit.h1_ddp.training.v2`，不能把旧 B1 恢复点直接当作四卡续训。

程序异常保留 rank 错误与当前评估恢复点，以失败状态退出。对同一 plan 使用
`worker --resume-incomplete` 或上面的 Slurm/本地入口续跑；已提交的权重/样例结果复用，
未完成部分补齐。训练权重、模式与随机数在评估后恢复。不要改正在运行的源码。

每个 run 的主要输出：

- `metrics.jsonl`：更新数、累计窗口、LR、loss、裁剪前梯度和各卡显存。
- `monitor/`、`evaluation_records/`：预测、逐样例结果与原子评估记录。
- `selection.json`、`checkpoint_inventory.json`：全程物理最佳及最后 checkpoint。
- `status.json`：实际停止更新、`endpoint_scores`、`schedule_complete` 与 GPU-hours。
- campaign `launcher_logs/`：各次命令、日志、退出码；失败证据保留。

`python -m graph_dit.campaign leaderboard --plan "$PLAN_JSON"` 查看汇总；
`python -m graph_dit.report --help` 提供曲线/报告入口。比较三种日程时同时报告 250k
窗口端点分数与全程最优分数，并标注日程是否走完；选择结论限于该 Validation 子集。

## 当前验收（2026-09-12）

本地 12 组计划生成成功；CPU 真实 H1 双进程 Gloo 通过了不同图尺寸下的独立全局
梯度参考、EMA 窗口换算、模型/EMA/Adam/每卡 RNG 精确恢复、评估故障续跑、无重复计数
和独立 checkpoint 读取。旧单卡 smoke 与早停恢复检查也通过。

```bash
python -m graph_dit.smoke --output-dir runs/cpu_fixture_new
python -m graph_dit.ddp_acceptance --fixtures runs/cpu_fixture_new --output-dir runs/cpu_ddp_new
```

目标四卡 CUDA/NCCL、d16/d24 最大图显存、吞吐和完整物理 Validation 尚待实际执行。
本次源码交付没有启动正式训练。
