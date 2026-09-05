# 多卡任务分发与恢复

## 一卡一实验

每张卡对应一个独立H1/B1进程。使用单机队列或Slurm array；同一个配置不使用`torchrun`或DDP。传入`WORLD_SIZE>1`会被拒绝，避免把有效batch变成卡数。

数据、共享表示和各run目录需在执行节点可见。代码与配置在整次campaign期间保持冻结；训练程序会复制所用Python源码，resume逐文件检查，拒绝代码或配方在途中变化。准备表示可以在一张卡上先完成；模型间共享文件，运行间不共享优化器或checkpoint。

默认计划现为100k/125k/150k首次十倍快降。旧版已有作业使用原plan和`ad6ada9272c4aac92327454347fbf68879876a78`源码继续，新版计划写到`campaigns/rapid_screen`；不在正在使用的checkout中切换日程。

## Slurm示例

在仓库根目录、激活好Python环境后提交。partition、account、时限、内存和并行额度按真实集群填写；脚本声明每个array task一张GPU。Slurm负责`CUDA_VISIBLE_DEVICES`。

```bash
export PLAN="$PWD/campaigns/rapid_screen/plan.json"
export DATA_DIR="/shared/cylinderflow/stride8"
export ARTIFACTS="/shared/cylinderflow/graph_dit_representation"
export PYTHON_BIN="$(command -v python)"
export REPO_ROOT="$PWD"

# 54个候选，最多同时运行16个。可按配额把%16改为%54。
sbatch --array=0-53%16 --partition=YOUR_PARTITION --account=YOUR_ACCOUNT \
  --time=2-00:00:00 --mem=32G --export=ALL scripts/slurm_array.sh
```

脚本在Slurm复制到spool目录后仍从`REPO_ROOT`或`SLURM_SUBMIT_DIR`找到源码。默认CPU线程2，申请4个CPU；`--mem`只影响主机内存申请，GPU显存由卡型号决定。先用512×12配置和最大Train粗图跑`graph_dit.preflight`。该检查包含多次更新、非零attention梯度、两套EMA和完整64帧采样解码。

晋级后改`PLAN=.../campaigns/rapid_extended/plan.json`，6个作业使用`--array=0-5%6`；确认阶段3个作业使用`--array=0-2%3`。计划生成和晋级命令只写任务清单，提交命令才消耗集群资源。

## 单机队列

```bash
python -m graph_dit.campaign run-local --plan campaigns/rapid_screen/plan.json \
  --data-dir "$DATA_DIR" --artifacts "$ARTIFACTS" --gpus 0,1,2,3,4,5,6,7
```

每个GPU worker顺序领取任务；一个任务异常会留下exit/log，其他独立任务继续。它不根据利用率或显存主动占用额外设备。`--gpus`是本机物理CUDA ID或UUID，须属于你当前的资源分配；在Slurm作业里优先直接使用array入口。

## 中断恢复

训练入口支持同一run的明确恢复：

```bash
python -m graph_dit.train \
  --config campaigns/rapid_screen/configs/h1_w256_d8_lr3e-05_late_decay_drop125000_seed0.json \
  --data-dir "$DATA_DIR" --artifacts "$ARTIFACTS" \
  --output-dir campaigns/rapid_screen/runs/h1_w256_d8_lr3e-05_late_decay_drop125000_seed0 \
  --stage-end-updates 250000 --device cuda:0 --resume
```

同一个run使用OS文件锁，重复进程不会同时写checkpoint。配置、源码、表示/cache ID必须一致；raw、全部EMA、Adam、随机状态、sample cursor和固定LR计划一起恢复。扩展分配只改变`stage-end-updates`，不改变原config。

队列重发时加`--resume-incomplete`。Slurm重发受影响的array index时设置`RESUME_INCOMPLETE=1`；由调用者决定重发哪些任务，不安装自动重启服务。已完成所分配终点且配置一致的作业会被跳过，运行中的文件锁仍受保护。

每次尝试写新的`attempt_*/training.jsonl`及失败档案；最近有效恢复点原子替换，50k持久checkpoint完整保留。若评价在中途失败，已完成候选保留，下一次恢复补完剩余候选。数值异常留下failure state与traceback，不能称为成功收敛。

## 运行目录

```text
campaigns/rapid_screen/
  plan.json, configs/*.json       # 确切候选和预定分配
  launcher_logs/                  # 逐作业命令、设备分配、日志、exit
  runs/<candidate>/
    config.json, source/          # 不可变配方及Python源码快照
    recovery_latest.pt            # 全状态滚动恢复点
    checkpoints/update_*.pt       # 持久raw+EMA+Adam+RNG
    attempt_*/                    # 逐次日志、错误、启动环境
    monitor/*/predictions/*.npz   # 每个候选及sampling seed的完整流场
    candidates.jsonl             # 含失败计数的所有候选
    selection.json, status.json   # 明确选择及当前分配完成状态
```

Checkpoint、完整预测、日志与源码应保留至后续补评测结束。只需补评分或绘图时使用已有NPZ；不必重训。回传权限由双方协作约定决定，脚本不自动发送任何资料。
