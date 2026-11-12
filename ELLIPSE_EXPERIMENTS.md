# Ellipse 完整实验入口

本分支提供与 CylinderFlow、Airfoil 对齐的实验类别：完整模型、确定性 Transformer、GNN diffusion、注意力消融、采样预算、同硬件成本及物理场对比。当前交付为代码与静态检查，不含正式实验结果。

## 数据与共享表示

沿用 [ELLIPSE.md](ELLIPSE.md) 的固定协议：Train782/Validation100/Test100、原始高分辨率网格、文件帧 1,16,…,961、UVP、1+64 预测、dt=0.15 s、预测长度9.6 s。所有阶段只访问保留的65帧，归一化仅用 Train，未来所有节点均预测，不回填边界。

使用现有生产 CUDA 环境。先完成 VGAE 和 Train latent 缓存，六种动力学共享同一个选中 VGAE、归一化及表示身份。缓存不按动力学模型重复构建；换表示时必须更换缓存目录。保留现有数据下载和准备入口。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR=/shared/data/megaflow2d_ellipse_stride15
export RAW_DATA_DIR=/shared/data/megaflow2d
export AUTOENCODER=/shared/runs/ellipse_vgae/campaign/runs/w512_d4-4-2_c4_seed0/dit_autoencoder.pt
export ARTIFACTS_DIR=/shared/artifacts/ellipse_latents
export COHORT=/shared/runs/ellipse_experiments
bash scripts/ellipse_experiments.sh prepare --cohort "$COHORT"
```

`AUTOENCODER` 缺省时仍可使用原入口的 `VGAE_REPO` 和 `VGAE_RESULT_ROOT` 串联训练；共享准备建议明确指定已选导出。`prepare` 不训练新的动力学。

## 消融训练和恢复

```bash
bash scripts/ellipse_experiments.sh train --cohort "$COHORT" --variant deterministic --seed 0
bash scripts/ellipse_experiments.sh resume --cohort "$COHORT" --variant deterministic --seed 0
```

| variant | 改变的设计 | 保持的设置 |
| --- | --- | --- |
| h1 | 主模型，一跳邻域 | 512宽、24层、8头、四通道潜变量 |
| h2 | 两跳硬邻域 | 同容量、同训练目标 |
| full | 全连接时空注意力 | 同容量、同训练目标 |
| distance | 归一化欧氏加权图最短路偏置，alpha=1 | 同容量；与两跳硬掩码分别标注 |
| deterministic | 零未来查询、固定噪声等级0、一次预测；标准化未来 latent MSE | 同 VGAE、同 Transformer 可训练骨干与条件 |
| gnn_diffusion | 128宽、三级深度1/2/2的 GNN U-Net，时间帧按有序通道联合去噪 | 同 VGAE、初态条件、1000级线性扩散、噪声 MSE、DDIM |

确定性模型复用原 Transformer 的所有可训练层，噪声等级固定，目标改为未来绝对潜变量。GNN 模型沿用已有 `feature/graph-latent-diffusion-4gpu` 的去噪实现，通过适配器接入共享训练器。GNN 额外层级从当前粗图的边与坐标派生，保留节点顺序；不读取未来场。两种对照均使用后验均值表示。

每卡1个完整窗口，四卡 global batch4、FP32、250k更新、100万窗口；AdamW、warmup1000、cosine到250k、EMA按窗口约定累计。每1250更新保存恢复点，每12500更新保留完整 checkpoint。`--seed 0|1|2` 控制动力学初始化和训练随机流，结果放在 `COHORT/<variant>_seed<seed>`，配置保存在独立 `COHORT/configs`。VGAE固定，不随动力学 seed 重训。

全部使用完整 Validation100 选优。h1/h2/full/distance 保持原 S20、标签0/1/2分别评分的选择规则；确定性对照使用一次前向，GNN diffusion 使用 S6/K8物理场均值。均比较 raw、EMA0.999、EMA0.9999。Test不参与该训练器的自动选权。

新正式运行使用新的结果目录；`resume` 保留优化器、随机状态、采样游标、EMA、选择历史及源快照。旧源快照须在其原工作区恢复；不要向活动运行覆盖代码。正式训练默认250k，`--stage-end-updates`只用于显式分阶段停止或下述减量验证。

## 固定权重评测和曲线

```bash
bash scripts/ellipse_experiments.sh validate --cohort "$COHORT" --variant deterministic \
  --output-dir /shared/evaluation/ellipse_deterministic_val
bash scripts/ellipse_experiments.sh test --cohort "$COHORT" --variant deterministic \
  --output-dir /shared/evaluation/ellipse_deterministic_test
bash scripts/ellipse_experiments.sh curves --cohort "$COHORT" --variant deterministic \
  --output-dir /shared/evaluation/ellipse_deterministic_curves
```

`--data-dir`与`--artifacts`默认读取上述环境变量。`--run`可绑定已有主模型结果；必须与`--variant`和`--seed`匹配。每次评测需要新的`--output-dir`。默认读取已完成阶段的`selection.json`。显式权重用`--checkpoint /absolute/checkpoint.pt --weights raw|ema_0.999|ema_0.9999`，输出记录选择方式，不覆盖训练选优。

`curves`事先核对12.5k、25k、…、250k全部20个checkpoint，每个分别评测Validation100与Test100，固定EMA0.999。完整模型、确定性、GNN各40项，共120项；两种扩散模型S6/K8，确定性一次前向。曲线默认不另存每个成员，保留平均场、逐轨迹与逐帧指标。终点曲线和Validation选优结果分别报告。

原入口仍兼容，并允许附加评测参数：

```bash
RESULT_ROOT="$COHORT/h1_seed0" EVAL_DIR=/shared/evaluation/ellipse_h1_s6k1 \
  bash scripts/ellipse_4gpu.sh test --sampling-steps 6 --ensemble-size 1 --seeds 0
```

常规评测默认保留逐成员物理预测和指标，`--skip-members`关闭额外成员输出。逐成员记录与平均场指标分别保存；样本失败保留在`failures.json`，四卡合并检查100条轨迹和所有标签完整唯一。`inference_seconds`可能包含成员评分和写盘，正式延迟只使用独立benchmark结果。

## 采样扫描与成本

```bash
bash scripts/ellipse_experiments.sh scan --cohort "$COHORT" --variant h1 --split test \
  --output-dir /shared/evaluation/ellipse_h1_sampling_test
# 切换到同一生产硬件上的单卡分配，单进程测速。
export CUDA_VISIBLE_DEVICES=0
bash scripts/ellipse_experiments.sh benchmark-scan --cohort "$COHORT" --variant h1 \
  --output-dir /shared/evaluation/ellipse_h1_sampling_cost
bash scripts/ellipse_experiments.sh benchmark --cohort "$COHORT" --variant deterministic \
  --output-dir /shared/evaluation/ellipse_deterministic_cost
```

扫描先固定单个Validation选中checkpoint和权重类型；不随S/K重新选权。K1扫描S={2,3,4,6,8,10,20,30,40,50,75,100}，联合扫描S={6,20}、K={1,2,4,8,16}，重叠配置只执行一次，共20组。`--sampling-labels 0 1 2`可做独立采样重复，每个标签使用固定成员编号段，K取嵌套前缀。不要把采样标签当作训练seed。

benchmark固定Validation列表中`rint(linspace(0,99,24))`的24条，单卡FP32、batch1，每轨迹预热2次、计时3次，包含CPU初态到CPU完整预测、编码、采样、解码与平均，排除质量评分和输出写盘。记录真实设备、依赖、参数量、峰值/增量显存及失败。既有Validation100计时接口保持可用，报告区分24/100注册表；跨方法成本比较必须匹配硬件和运行时。

AROMA、Text2PDE、MGN和EAGLE分支提供`bash scripts/ellipse_sampling.sh quality|benchmark`。AROMA和Text2PDE扫描K={1,2,4,8,16}，成员使用同一嵌套池，分别解码到物理场后取均值。MGN/EAGLE只使用K1。原主表三个独立样本分别评分的入口保持不变。见各仓库`ELLIPSE.md`。

## 汇总与共享色标

`python -m graph_dit.ellipse_report --manifest comparison.json --output-dir report`读取已有结果，不运行模型。输入路径相对manifest所在目录解析：

```json
{
  "evaluation_roots": [
    {"dataset": "Ellipse", "method": "GLaDiT", "root": "ellipse_h1_curves", "kind": "curves"},
    {"dataset": "Ellipse", "method": "Transformer", "root": "ellipse_deterministic_curves", "kind": "curves"},
    {"dataset": "Ellipse", "method": "GNN diffusion", "root": "ellipse_gnn_curves", "kind": "curves"}
  ],
  "evaluations": [],
  "runs": [],
  "cost_records": [],
  "comparisons": []
}
```

CylinderFlow/Airfoil已有同格式结果可作为独立dataset加入；单个结果可放入`evaluations`，使用`dataset/method/summary`字段。输出全指标CSV、训练损失CSV/曲线、质量曲线和Markdown表；不把缺失结果写成0，缺项记录于`missing.json`。

`runs`条目使用`dataset/method/run`，读取GLaDiT各动力学的status、原始日志和硬件记录，计算训练阶段及端到端吞吐和卡时。基线/VGAE/准备阶段使用`cost_records`：`dataset/method/stage/summary/world_size/hardware`，指向原阶段`summary.json`或原pipeline日志旁的计时JSON；不同阶段单列，失败attempt保留，不重复累加累计resume状态。Text2PDE没有完整更新计时的阶段记录只报告实测墙钟和卡时，不推算缺失吞吐。

`comparisons`条目包含`fields:[{"method":"GLaDiT","file":"...npz"}, ...]`和可选`frames:[1,24,48,64]`。工具要求所有方法的轨迹、真值、网格和时间轴一致，自动输出共享全局色标的速度模和压力图。案例建议从固定参照的完整Test误差排序选中位及最差轨迹，连同预先指定案例一起保存；在manifest保留选择依据。逐帧、压力、涡量、动能、enstrophy和时间谱指标沿用物理网格评测器。

## 验证状态

本次只完成语法、配置、Shell和差异等静态检查。当前本地环境未核实与四卡生产环境等价，未运行模型、转换、训练、评测或软件测试。

确认生产硬件型号、四卡拓扑、CUDA/NCCL和依赖版本匹配后，只使用一个减量入口：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
VERIFICATION_ROOT=/shared/verification/ellipse_det VARIANT=deterministic \
  bash scripts/ellipse_verify.sh
```

它使用正式模型、数据和四卡代码路径，训练1更新并保存、验证，再从恢复点继续到2更新；选择规则和完整Validation100保持不变。耗时含完整评测。后续修复复用同一脚本，已有尝试记录保留。该目录不用于正式结果；验证通过仅证明实际执行到的路径。
