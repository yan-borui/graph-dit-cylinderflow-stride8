# CylinderFlow：H1 / H2 / Full 四卡消融

本交付固定9个任务：三种注意力 × 训练seed 0/1/2。每任务单机四张同型号、至少48GB显存的GPU，
全局batch4；每任务250,000窗口、62,500更新。默认串行使用一组四卡，九任务同时运行需36卡。
当前交付为代码和静态检查结果；目标四卡运行验收由下述同一个preflight入口完成。

## 1. 环境与共享目录

在Linux中激活已有CUDA/PyTorch环境。源实现参考PyTorch 2.10；依赖见`pyproject.toml`。
使用与后续正式训练完全相同的GPU、驱动、系统、Python、Torch、PyG、NumPy、SciPy和h5py完成验收。
本流程使用NCCL、FP32，关闭AMP/TF32；HDF5文件要求兼容的h5py/HDF5运行时。
若环境尚未安装本源码且依赖已经齐全，可执行`python -m pip install -e . --no-deps`。
入口会优先从当前源码目录导入，不自动安装或升级依赖。

从仓库根目录执行，以下路径替换为集群实际目录：

```bash
export PYTHON=/shared/envs/gladit/bin/python
export DATA_DIR=/shared/cylinderflow/stride8
export ARTIFACTS_DIR=/shared/cylinderflow/artifacts_epoch1180_ablation
export COHORT=/shared/experiments/cylinderflow_attention_ablation_v1
# 非Slurm环境：填写已经分配的四张卡；Slurm环境保留调度器原有mask。
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

数据为Train1000/Validation100，固定版本`8eae2c7a697e7d01f3b98f4d642ea476784df84a`：
[CylinderFlow stride8数据](https://huggingface.co/datasets/DingDong1921/mgn-cylinderflow-stride8-75frames)。
VGAE来自[epoch1180发布](https://huggingface.co/DingDong1921/cylinderflow-vgae-uvp-epoch1180)，固定版本
`f21e74636a850c571f9870be91c6f420d41aa4d6`，`dit_autoencoder.pt`为249,943,363字节，
representation_id=`d0fee50b-8a47-4652-b229-0e82e06ba2d7`。按原Validation24 UVP总loss选中的这一份表示保持冻结。

```bash
bash scripts/ablation_4gpu.sh prepare
```

缺失的数据和权重会自动下载。可设置`AUTOENCODER=/shared/weights/dit_autoencoder.pt`复用已下载文件。
缓存编码全部Train75帧、Train-only统计，训练只取stored0..64；Test不下载、不读取。
准备过程使用共享锁，完整缓存才发布到`ARTIFACTS_DIR`；失败暂存目录和错误记录保留。
离线节点应先在联网且有获配GPU的准备节点完成此步，再共享数据、权重和同一份cache。
请使用本入口；仓库历史`prepare.py`的默认权重属于旧表示。

## 2. 唯一四卡验收与单机运行

```bash
# 三种结构各一次，始终使用正式四卡、真实模型、真实最大Train图。
bash scripts/ablation_4gpu.sh preflight all

# 跑全部9个任务：默认逐个使用当前四卡；也可选择一个任务。
bash scripts/ablation_4gpu.sh train all
# bash scripts/ablation_4gpu.sh train h2_seed1

# 中断后：同一源码、环境、目录和四卡，从最近保存状态继续。
bash scripts/ablation_4gpu.sh resume h2_seed1

# 锁定选优，然后分别评价终点与全程最佳，并计时终点模型。
bash scripts/ablation_4gpu.sh evaluate all
```

preflight固定先4步保存、恢复到8步；验证raw/两套EMA、Adam、各rank RNG与窗口游标恢复。
每个保存点使用原Validation24注册表的前4条轨迹，每rank一条，3次采样×3套权重，共72段评价。
它记录实际显存、loss、梯度、参数变化、异常和退出码。训练启动要求对应结构seed0验收成功；
正式训练从新初始化开始。失败后保留原目录；修复并重做验收时使用新的`COHORT`。

九个任务共用`COHORT/environment.json`。硬件、依赖、单节点拓扑必须一致；源码与表示身份也写入结果。
启动日志保存GPU状态、拓扑、包版本、torchrun日志和退出码。已有正式run使用`resume`，不能以`train`覆盖。
`resume all`适合九个run目录均已建立的恢复；未开始的任务仍使用`train TASK`。
平台期不会提前结束训练，数值/系统异常按实际错误保留证据。

## 3. Slurm数组

导出上面的`PYTHON/DATA_DIR/ARTIFACTS_DIR/COHORT`，让Slurm设置`CUDA_VISIBLE_DEVICES`。
按集群要求在命令上增加`--partition`、`--account`、`--time`和GPU型号约束。

```bash
# 先验收三种结构；完成后查看各preflight/任务/acceptance.json。
sbatch --array=0-2%1 scripts/slurm_ablation_4gpu.sh preflight

# 验收通过后提交9个正式任务；%1为同时运行1个四卡任务。
sbatch --array=0-8%1 scripts/slurm_ablation_4gpu.sh train

# 例如获配12卡时用%3；全部36卡可用%9。
# sbatch --array=0-8%3 scripts/slurm_ablation_4gpu.sh train

# 训练结束后评价；也可用--dependency=afterok:训练数组JOBID排队。
sbatch --array=0-8%1 scripts/slurm_ablation_4gpu.sh evaluate

# 只恢复index 4（H2 seed1）：
sbatch --array=4 scripts/slurm_ablation_4gpu.sh resume
```

index顺序：0/1/2为seed0的H1/H2/Full，3/4/5为seed1，6/7/8为seed2。
每个数组元素始终申请单节点四卡；并发限制只决定同时调度多少个任务。

## 4. 固定配方与结果口径

模型统一width512、24层、8heads、MLP ratio4、latent4、condition512，共115,013,124个参数。
首帧条件联合生成未来64帧；stride8、dt0.08、跨度5.12秒。保留CylinderFlow既有边界回填及回填前诊断。
H1含自身和一跳邻居，H2含自身和两跳以内邻居，Full连到全部节点；三组均跨全部时间槽。
H1/H2使用同一精确邻域SDPA，Full使用全连接注意力。节点级hop矩阵仍保留，不宣称全流程稀疏存储。
三组统一逐block激活重计算（`use_reentrant=False`）、DDP梯度bucket复用，参数结构和初始化顺序相同。
训练seed确定相同初始化、每epoch样本排列和rank噪声序列；三个生成采样标签与训练seed分别记录。

| 设置 | 全局窗口 | optimizer更新 |
|---|---:|---:|
| 完整预算及cosine终点 | 250,000 | 62,500 |
| warmup | 4,000 | 1,000 |
| checkpoint / Validation24 | 每50,000 | 每12,500 |
| 恢复点 | 每5,000 | 每1,250 |
| 日志 | 每100 | 每25 |

AdamW峰值LR1e-4，cosine终点1e-7，betas(0.9,0.999)、eps1e-8、weight decay1e-6、clip1。
每卡batch1、累积1、global batch4；EMA0.999/0.9999按窗口计，每更新使用0.999^4和0.9999^4。
每轮Validation24×3采样×raw/两套EMA共216段，每任务五轮，共九任务9,720段监控评价。
完整有限候选按UV主指标严格选优；相同分数按较早更新及raw/EMA固定顺序打破平局。

主表：62,500更新的raw/EMA由Validation24选中后，完整Validation100×3采样评价。
补充：五轮中全程最佳同样先由Validation24锁定，再评价Validation100。两者相同时共享评价文件。
Validation100包含选优的24条轨迹，属于验证集结果；不能标为独立Test结果。
跨seed均值/样本标准差仅在三seed全部完整后显示；H1−H2、H1−Full差值逐训练seed配对。
保留压力、涡量、能量、时间谱、边界诊断、失败记录；三种方法的结论限于本次25万窗口预算。

推理计时使用固定Validation24，四rank各6条；每条2次warmup、3次测量。
计时范围为CPU物理首帧→编码→20步DDIM生成64帧→解码/边界回填→CPU物理UVP。
模型载入、文件IO、指标计算、保存和渲染不计入延迟。报告每轨迹延迟和各rank显存；并行评价不把延迟除以4。
训练GPU小时、物理评价GPU小时、四卡实际占用时间、吞吐和各卡峰值显存另行记录。

## 5. 收集报告与保留结果

```bash
bash scripts/ablation_4gpu.sh report
# 连同共享色标对照视频：
MOVIES=1 bash scripts/ablation_4gpu.sh report
```

`COHORT/report/`输出`report.md`、`report.json`、`runs.csv`、`aggregate.csv`、`paired_seeds.csv`、
`paired_summary.csv`、`monitor_curves.csv`和`training_curves.png`；包括全部9个任务的状态与失败项。
`movies/`显示共同中位、p90、最难及H1相对对照误差差最大的案例，固定训练seed0/采样标签0。
案例难度由全部九个终点结果共同排序；每幅对照使用相同色标，选择规则写入`visual_cases.json`。

原始证据位于`runs/TASK/`与`evaluation/TASK/`，包括全部checkpoint、raw/EMA选择记录、
逐轨迹指标和物理预测；`launcher/`保存启动环境及退出状态。向合作者返回report目录及各任务
`status.json`，并保留完整cohort供复核。按模型、两套EMA及Adam两套moment的五份FP32状态估算，
每个完整checkpoint约2.3GB，九任务五个固定点加恢复点约125GB，此外还有验收、预测与日志；
请为完整结果预留数百GB空间，程序不自动删除证据。

## 6. 本地交付验证边界

本轮仅执行源码语法、配置一致性、Shell语法、静态lint及Git空白检查，并核对源码包逐文件内容。
没有执行模型、CPU替代测试、单卡替代测试或目标四卡验收。实际四卡容量、NCCL、恢复、数值行为
和正式指标由目标环境运行记录给出；没有为本次消融编造结果或耗时预测。

源码包基于`4901f33`，新分支为`feature/attention-ablation-4gpu`。重新生成交付包：

```bash
python scripts/package_ablation.py --output /shared/handoff/cylinderflow_attention_ablation_4gpu.zip
```

压缩包仅含运行源码、九配置、启动器、协议和说明；数据、权重由固定发布地址获取。
