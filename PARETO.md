# CylinderFlow 同机推理测速与绘图

本入口支持任意兼容的已选DiT权重。此次交接固定550k / EMA0.9999，
S=6、20，K=1、2、4、8、16；MGN/EAGLE/AROMA/Text2PDE各保留一个原配置点。
所有质量和耗时重新在同一机器生成，使用同一个新的`campaign-id`，Test保持封存。

## 获取各仓库代码与权重

本页是统一交接入口。测速脚本在各自代码仓库；汇总和绘图在本 DiT 仓库。
Hugging Face 提供固定版本的推理权重、配套 VGAE 和表示配置。

| 方法 | 仓库与分支 | 运行说明 |
| --- | --- | --- |
| GLaDiT | [graph-dit-cylinderflow-stride8 / main](https://github.com/yan-borui/graph-dit-cylinderflow-stride8) | 本页 |
| MGN | [meshgraphnets-cylinderflow-stride8 / cylinderflow-stride8](https://github.com/yan-borui/meshgraphnets-cylinderflow-stride8/tree/cylinderflow-stride8) | [PARETO.md](https://github.com/yan-borui/meshgraphnets-cylinderflow-stride8/blob/cylinderflow-stride8/PARETO.md) |
| EAGLE | [eagle-cylinderflow-stride8 / cylinderflow-stride8](https://github.com/yan-borui/eagle-cylinderflow-stride8/tree/cylinderflow-stride8) | [PARETO.md](https://github.com/yan-borui/eagle-cylinderflow-stride8/blob/cylinderflow-stride8/PARETO.md) |
| AROMA | [aroma-cylinderflow-stride8 / cylinderflow-stride8](https://github.com/yan-borui/aroma-cylinderflow-stride8/tree/cylinderflow-stride8) | [PARETO.md](https://github.com/yan-borui/aroma-cylinderflow-stride8/blob/cylinderflow-stride8/PARETO.md) |
| Text2PDE | [text2pde-multigeometry-1plus64 / feature/cylinderflow-stride8-1plus64](https://github.com/yan-borui/text2pde-multigeometry-1plus64/tree/feature/cylinderflow-stride8-1plus64) | [PARETO.md](https://github.com/yan-borui/text2pde-multigeometry-1plus64/blob/feature/cylinderflow-stride8-1plus64/PARETO.md) |

已有仓库请在对应分支执行 `git pull --ff-only`，有本地改动时先保留自己的工作。
首次使用可按表中分支 `git clone --branch <分支> <仓库URL>`，只获取所需方法。
在本 DiT 仓库根目录下载权重：

```bash
python scripts/download_pareto_weights.py --output-dir /shared/gladit_inference
```

下载器仅用 Python 标准库，固定模型 revision 为
`31183037b16ab2e6dba7122dce019c1dabd6297a`，不下载重复源码包。
四个 baseline 沿用既有已选 checkpoint、配置、绑定的 AE 和 prepared 数据，具体参数见各自说明。

## 环境与输入

复用各仓库既有CUDA环境，不升级训练依赖。为五个仓库分配同一张GPU，串行运行。
对齐CPU/GPU、Python、PyTorch、NumPy、CUDA/cuDNN和线程设置；汇总拒绝混用不同环境、
数据版本或campaign。依赖不匹配时保留报告并先协调环境，不加入旧5090结果。

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export CAMPAIGN=cylinderflow_same_machine_20260920
export DATA_DIR=/shared/cylinderflow_stride8
export WEIGHTS=/shared/gladit_inference
```

`DATA_DIR`包含既有75帧HDF5及`cylinderflow_stride8_75frames_manifest.json`。
数据为`DingDong1921/mgn-cylinderflow-stride8-75frames`，revision
`8eae2c7a697e7d01f3b98f4d642ea476784df84a`，归一化沿用Train统计。

模型仓库为`DingDong1921/gladit-cylinderflow-550k-ema09999`；固定下载revision和命令见本页上方。
从本仓库根目录运行；推理包包含`dit_ema.pt`和`artifacts/`，不依赖训练latent
缓存。训练入口仍检查完整缓存。依赖沿用仓库`pyproject.toml`和既有CUDA安装说明。

## DiT：生成十个点

```bash
python -m graph_dit.pareto_run \
  --checkpoint "$WEIGHTS/dit_ema.pt" --weights ema_0.9999 \
  --artifacts "$WEIGHTS/artifacts" --data-dir "$DATA_DIR" \
  --sampling-steps 6 20 --ensemble-sizes 1 2 4 8 16 \
  --campaign-id "$CAMPAIGN" --device cuda:0 --threads 2 \
  --output-dir "/shared/$CAMPAIGN/gladit"
```

`python -m graph_dit.benchmark`也接受相同参数。原训练目录可用`--run "$RUN"`替代
checkpoint/weights参数，从选优记录加载，支持暂停状态，不更新训练选优结果。

每个S/K点独立计时：完整预测K次，再对物理UVP求平均，不在latent中平均。K使用标签0至K−1，
两个S共享seed规则。计时重复不增加K。`--mode timing`或`--mode quality`用于单项诊断，
默认`all`生成可直接汇总的配对点；正式图使用`all`输出。

测速固定Validation24，每条预热2次、计时3次，FP32、batch1、无TF32/autocast。
范围为CPU首帧/静态网格已就绪至CPU物理输出，包含完整64帧、VGAE、传输、边界处理和集成平均；
模型加载、磁盘I/O、静态图准备、评分及保存不计入。所有方法遵循已有共同协议。
质量固定Validation100，每点保存100份平均场预测、逐轨迹指标和采样标签/seed规则；
不额外存储重复成员池。各K的预测独立生成，避免复用池漏计真实成本。

## 四个baseline

各仓库根目录`pareto_run.py`和`PARETO.md`给出命令。使用已有checkpoint、AE及prepared目录，
先运行原benchmark，再运行正式Validation100。MGN/EAGLE单份预测；AROMA/Text2PDE
沿用三个独立单样本分数均值，不取三份物理场平均。baseline横轴为一次预测成本，纵轴为
原单样本质量估计。旧`performance`汇总器保持兼容，本任务的新汇总器支持同方法多点。

## 汇总和绘图

```bash
python -m graph_dit.pareto_plot \
  --inputs "/shared/$CAMPAIGN/gladit" "/shared/$CAMPAIGN/mgn" \
    "/shared/$CAMPAIGN/eagle" "/shared/$CAMPAIGN/aroma" "/shared/$CAMPAIGN/text2pde" \
  --campaign-id "$CAMPAIGN" --output-dir "/shared/$CAMPAIGN/figure"
```

输出`pareto.csv`、`pareto.png/.pdf/.svg`、`issues.json`。横轴为64帧端到端秒数（对数轴），
纵轴为Validation100平均UV相对RMSE；S=6/20两条线按K连接并标注K，baseline各单点。
连线表示配置变化，所有实测点保留。

`--metric vorticity_rmse`等切换已有标量，`--title`设置标题，`--collect-only`只生成CSV。
缺失/失败点默认阻止正式出图；`--allow-partial`产生标有PARTIAL的部分图及缺口清单。
不同campaign、数据和执行环境始终拒绝混合，不插值。输出目录必须不存在。

## 验证与日志

交接前仅做静态检查、真实归档绘图渲染检查和权重内容检查，没有用本地CPU替代GPU运行。
学长在最终环境中保留日志、exit.json、原始计时和预测。如需先缩小工作量，可先指定单一S/K，
仍保留正式24/100轨迹、完整模型、单卡、64帧和FP32，再用新输出目录运行完整网格。

## 发布范围与推理实现

此次发布包含 550k UVP-c4 表示的加载支持、仅推理时的缓存豁免，以及已交付源码使用的
H1 neighbor attention 实现与其可选 flex 模块；checkpoint 参数和默认推理实现保持配套。
原训练启动器、调参配置和训练预算不随本次交接修改。
静态检查不能替代最终 CUDA 环境中的端到端验证。
