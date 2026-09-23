# Airfoil attention 消融

三组实验分别使用 one-hop、two-hop 和 full attention。三组均以 seed0 训练，
共用同一份 Airfoil VGAE 导出和 Train latent cache，模型及训练配方沿用
[Airfoil 四卡训练](AIRFOIL.md)。每组四卡各 B1、全局 B4，FP32，width512、depth24、
heads8、latent4、condition512，训练 250,000 次更新、1,000,000 个窗口。
完整 cosine、warmup、EMA、验证频率和选优规则保持一致。

## 运行与恢复

在独立 checkout 中使用本分支，保留正在训练的主实验源码。从仓库根目录运行。
激活集群原有环境，填写已选中的 Airfoil VGAE 导出、
共享缓存及独立实验目录。Slurm 作业保留调度器分配的 GPU mask。

```bash
export PYTHON=python
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAW_DATA_DIR=/data/datasets/meshgraphnets/airfoil
export DATA_DIR=${RAW_DATA_DIR}_uvp_stride8
export AUTOENCODER=/shared/runs/airfoil_vgae_seed0/campaign/runs/w512_d4-4-2_c4_seed0/dit_autoencoder.pt
export ARTIFACTS_DIR=/shared/artifacts/airfoil_vgae_seed0
export COHORT=/shared/runs/airfoil_attention

# 自动完成全量 Train/Validation 数据准备及共享表示验证，然后训练。
bash scripts/airfoil_ablation_4gpu.sh train h1
bash scripts/airfoil_ablation_4gpu.sh train h2
bash scripts/airfoil_ablation_4gpu.sh train full

# 中断后保持配置、源码、表示、设备拓扑和环境，恢复对应任务。
bash scripts/airfoil_ablation_4gpu.sh resume h2
```

三条训练命令按顺序使用同一组四卡，每组有独立结果目录。入口锁定本组实验，
防止同时重复启动。既有训练目录使用 resume；失败日志和恢复点继续保留。
需要单独提前准备缓存时运行 `bash scripts/airfoil_ablation_4gpu.sh prepare h1`。
请在独立获配的四卡资源上运行；既有 Airfoil 主实验继续按原入口管理。

H1 覆盖自身及一跳邻居，H2 覆盖两跳以内节点，Full 覆盖全部节点，均连接全部时间槽。
三种配置只改变 attention，专用协议校验锁定其他字段。H1、H2 均按空间邻域大小分组，
收集允许邻居的全部时间槽，再调用 SDPA。H2 按图距离0、1、2取邻居，保持原两跳约束，
空间关系保存在节点级矩阵中。Full 使用全连接路径。实际容量和耗时由目标设备日志记录。
各组保持完整模型、四卡拓扑和精度。平台期保持完整训练预算。

## 结果汇总与回传

```bash
bash scripts/airfoil_ablation_4gpu.sh report
```

汇总输出 `attention_summary.csv` 和 `attention_summary.json`，保留三组状态、完成更新数、
选中 checkpoint、raw/EMA 权重、表示身份及原始 Validation-24 UV 指标，
同时列出完整预算终点 raw 和两套 EMA 的原始分数。缺失或失败任务保留在表中，
配置不匹配时指标留空。训练中的结果随最新记录更新，达到完整预算后用于最终比较。

选优沿用未来 64 帧面积加权 UV relative RMSE，每条轨迹对三次独立采样评分后平均，
再对 Validation-24 等权平均。每次采样固定 20 步，比较 raw 和两套 EMA，
仅完整有效候选严格改善时更新选中结果。Test 保持封存。

请回传汇总表，以及每组的 config、status、selection、candidates、evaluation_records、
monitor 物理评价目录和 launcher 日志。选中的权重及 recovery_latest.pt 保留在集群，
供结果核对和后续画图使用。训练日志目录与结果目录同级，以 `_launcher_` 命名。

## 验证状态与 CylinderFlow 交接

本交接执行 Python/JSON 静态语法、三配置差异和 shell 语法检查。
目标四卡 Linux/NCCL 环境尚待集群核验；训练、恢复及显存容量的运行证据由集群提供。

CylinderFlow 继续使用现有
[四卡消融分支及说明](https://github.com/yan-borui/graph-dit-cylinderflow-stride8/blob/feature/attention-ablation-4gpu/ABLATION_4GPU.md)。
该交接已包含准备、唯一四卡验收、训练、恢复、评价、汇总及 Slurm 数组入口。
其每任务预算为 62,500 次更新、250,000 个窗口，三训练种子共九任务；
Airfoil 本交接遵循自己的完整四卡训练预算。
