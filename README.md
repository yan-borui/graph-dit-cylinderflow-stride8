# Graph DiT · CylinderFlow · H1 / B1

给定一帧真实流场，在 VGAE 的粗图上联合生成未来 64 帧。这个独立仓库交付当前 Graph DiT、冻结表示、训练配方、并行调参、恢复、共同评价和报告脚本。**固定 H1 attention、microbatch=1、梯度累积=1、effective batch=1；多卡用于独立实验并行。**

本仓库提供可运行的搜索方案。`configs/base.json` 是起跑配方；经过完整搜索和独立训练复核以后，`freeze` 才会产生 `locked_recipe.json`。当前没有把尚未执行的候选称为最佳模型。

## 快速开始

先按 [ENVIRONMENT.md](ENVIRONMENT.md) 建立环境。全部命令在仓库根目录执行；需要 Linux CPU 或 CUDA GPU。数据、缓存和运行目录都可放在集群共享盘上。

```bash
git clone https://github.com/yan-borui/graph-dit-cylinderflow-stride8.git
cd graph-dit-cylinderflow-stride8
python -m pip install -e .

# 软件交付验收：合成数据、实际 VGAE/DiT 更新、恢复、64 帧推理与报告。
CUDA_VISIBLE_DEVICES='' python -m graph_dit.smoke --output-dir runs/cpu_acceptance --movies

# 只下载固定版本的公开 Train/Validation 文件。原始 HDF5 约 1.77 GB。
python -m graph_dit.fetch --output-dir data/stride8

# 复用当前选定的 VGAE（epoch 930，约 9.59 MB）；整个搜索只准备一次表示。
python -m graph_dit.representation fetch-ae --output artifacts/vgae_stride8_epoch930.pt
python -m graph_dit.representation prepare \
  --data-dir data/stride8 --autoencoder artifacts/vgae_stride8_epoch930.pt \
  --output-dir artifacts/shared --device cuda:0

# 生成 54 份明确的配置与任务清单，不启动训练。
python -m graph_dit.campaign plan --output-dir campaigns/screen

# 先验收目标设备及计划中的最大容量。
python -m graph_dit.preflight \
  --config campaigns/screen/configs/h1_w512_d12_lr0.0001_constant_seed0.json \
  --data-dir data/stride8 --artifacts artifacts/shared \
  --output-dir runs/preflight_w512_d12 --device cuda:0

# 例如使用已经分配给你的 8 张 GPU；每卡同时只有一个训练进程。
python -m graph_dit.campaign run-local --plan campaigns/screen/plan.json \
  --data-dir data/stride8 --artifacts artifacts/shared --gpus 0,1,2,3,4,5,6,7
```

拥有 64 张卡时可用 Slurm array 分发到多个节点，见 [CLUSTER.md](CLUSTER.md)。单机队列的 `--gpus` 只填写本机且已分配给你的设备。目标 GPU 的最大网格显存、精度和吞吐验收仍需在目标环境取得。

## 调参顺序

| 步骤 | 工作 | 固定分配 |
| --- | --- | --- |
| Screen | LR `1e-5 / 3e-5 / 1e-4` × cosine / late_decay / constant × width `256 / 384 / 512` × depth `8 / 12` | 54 作业，seed 0，各 250,000 updates |
| Extend | Validation-24 选择 6 个候选，恢复同一原始状态与调度 | 续至各 1,000,000 updates |
| Freeze / confirm | 锁结构、LR、调度及 raw/EMA 规则；独立 training seeds `101 / 102 / 103` | 各 1,000,000 updates，全部报告 |
| Full Validation | 每个独立训练运行选定一个 checkpoint，评价 Validation-100 × sampling seeds 0/1/2 | 每模型 300 clips |

EMA `0.999 / 0.9999` 在每次训练中同时维护，和 raw 在相同 checkpoint、相同采样噪声下比较。学习率计划从一开始就定义到 100 万更新；25 万仅是第一段分配，续训不会重启 warmup 或修改前半程日程。具体公式、容量含义、训练时长的选择及预算见 [RECIPE.md](RECIPE.md)。

```bash
# 首轮所有计划作业结束（包括留下失败记录的作业）后继续。
python -m graph_dit.campaign promote --plan campaigns/screen/plan.json \
  --top-k 6 --stage-end-updates 1000000 --output-dir campaigns/extended
python -m graph_dit.campaign run-local --plan campaigns/extended/plan.json \
  --data-dir data/stride8 --artifacts artifacts/shared --gpus 0,1,2,3,4,5

python -m graph_dit.campaign freeze --plan campaigns/extended/plan.json \
  --seeds 101,102,103 --output-dir campaigns/confirm
python -m graph_dit.campaign run-local --plan campaigns/confirm/plan.json \
  --data-dir data/stride8 --artifacts artifacts/shared --gpus 0,1,2
```

## 学长需要保留和回传什么

完整 checkpoint、优化器/EMA、源码快照、逐条预测、日志与失败记录保存在训练环境。生成可读曲线、配置卡、候选表、配对 GIF/MP4，并按双方允许的范围回传精简结果；没有自动上传或发送数据的功能。

```bash
python -m graph_dit.report --plan campaigns/extended/plan.json --output-dir reports/candidates
python -m graph_dit.report --run campaigns/confirm/runs/confirmation_seed101 \
  --output-dir reports/seed101 --movies
python -m graph_dit.evaluate --run campaigns/confirm/runs/confirmation_seed101 \
  --data-dir data/stride8 --artifacts artifacts/shared \
  --scope validation --output-dir runs/validation_seed101 --device cuda:0
```

对 seed102/103 同样执行完整评价，并保留三次训练结果。取同一 checkpoint 的三条采样种子属于采样随机性，不能充当三次训练复现。完整结果表、跨方法配对视频和测速命令见 [EVALUATION.md](EVALUATION.md)。

## 数据与四个 baseline

公开数据固定为 [CylinderFlow stride-8 75 frames](https://huggingface.co/datasets/DingDong1921/mgn-cylinderflow-stride8-75frames/tree/8eae2c7a697e7d01f3b98f4d642ea476784df84a)：Train 1000 / Validation 100 trajectories。VGAE 使用全部 75 帧；DiT 每轨迹固定 stored `0..64`，输入 frame 0，评价 future `1..64`，`dt=0.08`，物理跨度 `5.12`。Test 封存，所有入口只接受 Train/Validation。

共同物理指标、预测 NPZ schema、轨迹聚合、checkpoint 排序和独立测速模块与下列已发布版本对齐：

- [MeshGraphNets](https://github.com/yan-borui/meshgraphnets-cylinderflow-stride8/tree/7ef6fff8068424e195438fafe84ff7bc170b305a)
- [EAGLE](https://github.com/yan-borui/eagle-cylinderflow-stride8/tree/54054905feacba1eb2e48813432ebb9fb0ec3640)
- [AROMA](https://github.com/yan-borui/aroma-cylinderflow-stride8/tree/89011a2a3fc3264ebb102b1510d822328c00bf1c)
- [Text2PDE](https://github.com/yan-borui/text2pde-multigeometry-1plus64/tree/471b0a12536088a8528fe853b39d3b9d252e1112)

各方法保留自己的表示、原生训练单元、优化器及预算。Graph DiT 的调参成本单独披露；完整模型性能比较和纯 attention 因果消融是不同实验。本次任务只调 H1。

验收范围和实际执行结果见 [ACCEPTANCE.json](ACCEPTANCE.json)。源代码来源及 Apache-2.0 许可见 [NOTICE.md](NOTICE.md)、[LICENSE](LICENSE)。
