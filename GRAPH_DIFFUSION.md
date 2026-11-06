# 图潜空间扩散

该模型从一个观测流场和静态网格联合生成未来 64 帧 UVP。两种模型共享投稿 VGAE 表示：GLaDiT 在粗图上用 Transformer 去噪，本模型用三级消息传递 GNN U-Net 去噪。比较时应同时给出各自的参数量、有效 batch、训练窗口数、采样预算和端到端成本。

## 训练目标与采样

- 固定每个数据集投稿的 VGAE、物理归一化及 latent 归一化。新模型只读取存储帧 0–64（原始帧 0, 8, …, 512）；绑定的历史表示和统计量可包含 Train75 暴露。缓存身份和边界处理沿用[表示与数据说明](DETERMINISTIC.md)。
- 每个粗图节点输入首帧 latent、静态条件及该节点 64 个带噪未来 latent。时间顺序保存在固定的输出通道中。U-Net 沿图下采样、传递信息、上采样，宽度 128，三级深度为 1、2、2。输出层同时读取图特征与按噪声等级缩放的带噪输入，让 256 个噪声分量各自具有直接的预测路径。
- 训练时每个窗口抽取一个共享噪声等级，对未来 64 帧各节点各通道加独立高斯噪声，以噪声 MSE 监督。1,000 级 DDPM 使用线性 beta（0.0001–0.02）。首帧作为干净条件。推理从整段高斯噪声开始，使用 DDIM（η=0）联合更新，得到标准化未来 latent 后通过同一 VGAE 解码。
- 默认 S6/K8，即每个成员做 6 次去噪，对 8 个成员分别解码再取物理场均值，一条轨迹共做 48 次去噪网络前向。CylinderFlow 回填入口、壁面的已知速度，Airfoil 预测全部节点。去噪算法的步数与 64 个物理时刻分别计数。

## 准备和启动

使用现有的 Linux CUDA/NCCL 环境，从仓库根目录运行。两个数据集各用自己的投稿 `PAPER_ARTIFACTS` 目录，包含 `artifact.json` 和匹配的 `autoencoder.pt` 或 `dit_autoencoder.pt`。CylinderFlow 的表示和上游资产 ID 已固定。Airfoil 须将同一投稿选中 checkpoint 记录的两个 ID 设置为 `PAPER_REPRESENTATION_ID`、`PAPER_ARTIFACT_ID`。

```bash
export WORK_DIR=/shared/runs/rebuttal
export PAPER_ARTIFACTS=/shared/paper/cylinderflow/artifacts
bash scripts/graph_latent_diffusion_4gpu.sh cylinderflow train
bash scripts/graph_latent_diffusion_4gpu.sh cylinderflow resume
```

Airfoil 用相同命令，将数据集参数改为 `airfoil`，并切换到相应的导出和两个 ID。`train` 自动准备输入，或先运行 `prepare`。已有确定性对照缓存时，可指定相同的 `DATA_DIR` 和 `CACHE_DIR`，入口会核对来源。两种对照的 `RESULT_ROOT` 必须分别指定或使用各自的默认值。恢复同一次运行使用同一份代码、缓存与结果目录；此前版本的确定性运行在建立它的原分支恢复。

每个 rank 每步取一条完整轨迹，四卡合计 batch 4，FP32，250,000 次更新，共 1,000,000 个窗口。AdamW 峰值学习率为 0.0001，warmup 1,000 次更新，随后余弦衰减。按窗口换算 EMA 系数，每 1,250 次更新保存恢复状态，每 12,500 次更新在固定 Validation24 上用解码后的全窗速度相对误差选择 raw/EMA。选优使用 S6/K8 及固定成员种子。训练只访问 Train、Validation。

## 完整评价与测速

```bash
# 沿用启动时的 WORK_DIR、缓存和投稿身份设置
bash scripts/graph_latent_diffusion_4gpu.sh cylinderflow eval
SPLIT=test TEST_DATA_DIR=/shared/data/cylinderflow-test \
  bash scripts/graph_latent_diffusion_4gpu.sh cylinderflow eval
ENSEMBLE_SIZE=1 bash scripts/graph_latent_diffusion_4gpu.sh cylinderflow eval
bash scripts/graph_latent_diffusion_4gpu.sh cylinderflow benchmark
```

`eval` 读取 Validation 选定的权重，先从首帧预测，再读取未来真值，输出完整集上的逐轨迹指标及汇总。`SAMPLING_STEPS` 和 `ENSEMBLE_SIZE` 仅改变最终评价/测速的预算，用同一组权重即可比较 K1 与 K8。输出目录默认按 split、S、K 分开；重复同一配置用新的 `OUTPUT_DIR`。

`benchmark` 在单卡固定 Validation24 上，从 CPU 初始状态和预载的静态网格计到最终 CPU 物理场，包含编码、全部去噪与解码、边界处理和场平均。每轨迹预热两次、计时三次；报告给出硬件、秒数、显存和失败数。

在已核实与目标机等价的四卡分配中，可设置 `VERIFICATION_UPDATES=2` 和独立 `RESULT_ROOT` 运行同一训练与恢复入口，核验训练、Validation、保存和恢复。交付时已做静态检查，正式四卡执行与性能仍待目标机实测。
