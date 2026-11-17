# Temporal GNN diffusion 宽度阶梯

在 `feature/graph-latent-diffusion-4gpu` 分支运行 CylinderFlow、Airfoil 的宽度 256、512、1024 对照。每项从头训练，各占一组四卡。原宽度 128 的权重及 K8 结果继续保留。

| 隐藏宽度 | 可训练参数量 | 相对原 GNN |
| ---: | ---: | ---: |
| 128，已有 | 2,480,640 | 1.00× |
| 256 | 9,417,472 | 3.80× |
| 512 | 36,857,088 | 14.86× |
| 1024 | 146,000,128 | 58.86× |

参数量按现有构造函数静态计算，正式运行的 `attempt_*/launch.json` 记录实际数量。GLaDiT 骨干为 115,013,124 参数。三档均固定三级网格、深度 `(1,2,2)`、latent 维数 4、条件维数 512、未来 64 帧；增加隐藏通道及对应的噪声步嵌入宽度。

## 固定训练与评价协议

- 沿用各数据集原投稿 VGAE、归一化、首帧条件、缓存和边界语义。CylinderFlow 回填已知入口/壁面速度；Airfoil 预测全部节点。
- 四卡 DDP，每卡一条完整轨迹，有效 batch 4，训练 seed 0，250,000 更新、1,000,000 个窗口。FP32、TF32 关闭。
- AdamW，峰值学习率 `1e-4`、下限 `1e-7`，warmup 1000，250K 余弦调度，weight decay `1e-6`、梯度裁剪 1.0。三档采用同一配方。
- 新阶梯只维护 EMA 0.999，保持按窗口换算衰减的惯例。每 5000 更新保存恢复状态；每 12500 更新保留 checkpoint，并在固定 Validation24 上用 EMA 0.999、S6/K1、标签 0 监控。`selection.json` 如实记录该监控下的最佳点。
- **正式比较固定终点 250K / EMA 0.999**，不读取或改写选权记录。Validation100/Test100 各做 S6、标签 0/1/2 三次独立 K1，逐次生成、解码、评分，先平均同轨迹的三次分数，再对 100 条轨迹等权平均。保存实际种子，任一次失败即标记结果不完整。

原 128 入口仍使用原 raw/双 EMA、Validation24 S6/K8 选权和 1250 更新恢复间隔。新阶梯的监控与保存频率变化写入各自的 `config.json`。最终对照的权重、采样与评分协议一致。

## 六项启动命令

使用集群既有的生产 Linux CUDA/NCCL 环境；每条命令分别提交到一组四卡分配，沿用调度器提供的 `CUDA_VISIBLE_DEVICES`。`run` 在训练成功后接续完整 Validation100 和 Test100 评测。最终评测使用这组卡中的第一张；其余卡不会自动启动其他实验。

设置一个新的共享结果根目录，并复用原始数据与缓存。共享缓存的准备沿用原入口的排他锁和身份核查。

```bash
export WORK_DIR=/shared/runs/gnn_width_ladder_20261117
# PYTHON 可指向现有生产环境的 Python；默认使用当前 python。
```

先在 CylinderFlow 作业环境中设置其资产路径：

```bash
export DATA_DIR=/shared/cylinderflow/data65
export CACHE_DIR=/shared/cylinderflow/original-train-cache
export TEST_DATA_DIR=/shared/cylinderflow/test65
export PAPER_ARTIFACTS=/shared/cylinderflow/paper-artifacts

bash scripts/graph_diffusion_ladder_4gpu.sh cylinderflow 256 run
bash scripts/graph_diffusion_ladder_4gpu.sh cylinderflow 512 run
bash scripts/graph_diffusion_ladder_4gpu.sh cylinderflow 1024 run
```

再在 Airfoil 作业环境中设置其资产路径与原投稿身份：

```bash
export DATA_DIR=/shared/airfoil/data65
export CACHE_DIR=/shared/airfoil/original-train-cache
export TEST_DATA_DIR=/shared/airfoil/test65
export PAPER_ARTIFACTS=/shared/airfoil/paper-artifacts
export PAPER_REPRESENTATION_ID='<original-airfoil-representation-id>'
export PAPER_ARTIFACT_ID='<original-airfoil-artifact-id>'

bash scripts/graph_diffusion_ladder_4gpu.sh airfoil 256 run
bash scripts/graph_diffusion_ladder_4gpu.sh airfoil 512 run
bash scripts/graph_diffusion_ladder_4gpu.sh airfoil 1024 run
```

将示例路径和两个身份占位符换成上一轮的真实值。六条命令是六项独立作业；总卡数不足时由集群排队。默认结果目录分别为 `WORK_DIR/<dataset>/w<width>-seed0`，从而避免宽度间覆盖。

## 恢复、单独评测与进度

沿用同一代码快照、环境、路径和四卡配置恢复对应宽度：

```bash
bash scripts/graph_diffusion_ladder_4gpu.sh cylinderflow 1024 resume-run
```

`train` / `resume` 只训练；`evaluate` 只运行固定终点评测。评测会从 `status.json` 读取运行 UUID，再核对 checkpoint 的更新数、宽度、EMA、架构及数据身份。宽度记录同时用于原生选权评价与测速的模型重建。

```bash
bash scripts/graph_diffusion_ladder_4gpu.sh cylinderflow 1024 evaluate
bash scripts/graph_diffusion_ladder_4gpu.sh cylinderflow 1024 status
```

`status` 只读已有日志，报告最近最多 12500 更新的秒数/更新与剩余训练时间估计；样本不足一个监控周期时标为 provisional。估计不含最终完整评测、未来排队和未发生的故障，不触发暂停或终止。

默认完整评测目录为 `RUN/reports/matched_250k_s6_k1_r3/{validation,test}`。已有输出不会覆盖。若其中一个划分失败，保留它的记录，设置 `SPLIT=validation` 或 `SPLIT=test`、以及新的 `EVAL_OUTPUT_DIR` 后单独重评。`run` / `resume-run` 发生训练或评测错误时返回非零退出码。

已有训练的恢复使用它保存的 `source` 快照；本分支新增代码不会改动这些快照。原 128 checkpoint 可由本版固定 K1 评测入口读取，结果会补充架构及参数量元数据。

## 收集阶梯结果

十二项固定终点评测全部完成后运行：

```bash
python -m graph_latent_diffusion.ladder collect \
  --summaries "$WORK_DIR"/*/w*-seed0/reports/matched_250k_s6_k1_r3/*/summary.json \
  --output-dir "$WORK_DIR/comparison"
```

输出 `comparison.csv` 和 `comparison.md`，列出数据集、划分、宽度、参数量、UV 相对 RMSE、运行 UUID 和源结果。收集器要求三档 × 两数据集 × 两划分恰好完整，并核对原表示、总体、深度、参数量、250K/EMA 0.999、S6、三次独立 K1 和成功状态。

可用 `--references` 显式附加原 128 GNN 或 GLaDiT 的匹配 K1 汇总；每种参照均提供两个数据集和两个划分的四份结果。原 128 参照用本版 `graph_dit.matched_k1 evaluate --model gnn --expected-gnn-width 128 ...` 生成，命令的其余参数见 [MATCHED_K1.md](MATCHED_K1.md)。GLaDiT 参照复用原分支的匹配结果。K8、监控 Validation24、其他 checkpoint 或不完整结果均不会纳入最终表。

本次交付完成源码与 Shell 静态检查。四卡 H20 运行环境、实际速度、峰值显存及新模型结果需在集群执行后确认；未启动 GPU 作业。
