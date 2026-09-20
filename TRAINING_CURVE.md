# 给学长：四个 baseline 历史 checkpoint 的 UV 误差表

请在 **MGN、EAGLE、AROMA、Text2PDE 四个仓库各运行一次下面的收集脚本**。
使用之前训练留下的各个 checkpoint，脚本依次跑完整 Validation100，记录UV相对RMSE，
自动生成清晰的分页表格图片。跑完把图片发群里即可；GLaDiT这列由我在5090上跑好后补入。

本次沿用已有模型、数据及CUDA推理环境，只做历史权重评价。无需重新训练、下载DiT权重或运行测速脚本。
四个方法可以用各自可用的GPU；每个方法内部串行评价各checkpoint。

## 1. 更新四个仓库

| 方法 | 需要更新的分支 | 本仓库运行说明 |
| --- | --- | --- |
| MGN | `cylinderflow-stride8` | [MGN说明](https://github.com/yan-borui/meshgraphnets-cylinderflow-stride8/blob/cylinderflow-stride8/TRAINING_CURVE.md) |
| EAGLE | `cylinderflow-stride8` | [EAGLE说明](https://github.com/yan-borui/eagle-cylinderflow-stride8/blob/cylinderflow-stride8/TRAINING_CURVE.md) |
| AROMA | `cylinderflow-stride8` | [AROMA说明](https://github.com/yan-borui/aroma-cylinderflow-stride8/blob/cylinderflow-stride8/TRAINING_CURVE.md) |
| Text2PDE | `feature/cylinderflow-stride8-1plus64` | [Text2PDE说明](https://github.com/yan-borui/text2pde-multigeometry-1plus64/blob/feature/cylinderflow-stride8-1plus64/TRAINING_CURVE.md) |

在对应仓库、上述分支执行 `git pull --ff-only`。有本地修改先保留，勿覆盖正在训练的源码。
然后激活该仓库之前使用的CUDA环境，从仓库根目录运行。

## 2. 准备路径并运行

命令中的变量请设成机器上的**现有绝对路径**：

| 变量 | 填什么 |
| --- | --- |
| `RUN` | 一个已训练run的目录，内部有`checkpoints/`；也可直接填存放checkpoint的目录 |
| `CONFIG` | 该run使用的正式配置；MGN/EAGLE/AROMA为JSON，Text2PDE为LDM YAML |
| `DATASET`、`MANIFEST` | 原CylinderFlow数据HDF5与数据manifest（前三个仓库使用） |
| `PREPARED` | 原预处理输出目录，含归一化、图或latent等（前三个仓库使用） |
| `AE` | AROMA动力学或Text2PDE LDM绑定的已选AE checkpoint |
| `OUT` | 本方法新的结果目录，例如`/shared/uv_curve/mgn`；四方法分别保存 |

AROMA的RUN指向动力学训练目录，Text2PDE的RUN指向LDM训练目录。
Text2PDE数据位置沿用CONFIG中的设置。请在每个仓库重新设置相应变量，避免带入上一个方法的路径。

各方法先设置：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
```

**MGN、EAGLE：分别在各自仓库运行**

```bash
python training_curve.py --run-dir "$RUN" --config "$CONFIG" \
  --dataset "$DATASET" --manifest "$MANIFEST" --prepared "$PREPARED" \
  --output-dir "$OUT"
```

**AROMA：**

```bash
python training_curve.py --run-dir "$RUN" --config "$CONFIG" \
  --dataset "$DATASET" --manifest "$MANIFEST" --prepared "$PREPARED" \
  --ae-checkpoint "$AE" --output-dir "$OUT"
```

**Text2PDE：**

```bash
python training_curve.py --run-dir "$RUN" --config "$CONFIG" \
  --ae-checkpoint "$AE" --output-dir "$OUT"
```

脚本从checkpoint内部读取update并排序，逐个加载、评分，每完成一个点就更新表格。
保留原采样协议：MGN/EAGLE单次预测，AROMA/Text2PDE三个独立样本的分数平均。
重复运行原命令会跳过已完成点；失败或中断点在原命令后加`--retry-failed`重试。
具体错误在`OUT/update_*/attempt_*/evaluation.log`，排除原因在`OUT/status.json`。
如报残留`collector.lock`，先确认旧收集进程已结束，再移除锁并重试。

## 3. 跑完回传这些图片

**每个方法回传`OUT/pages.json`列出的全部`table_XX.png`，发原图即可。**
每页最多15行，图片标明方法、run、页码，数据只有两列：

| update × 4 | UV相对RMSE |
| ---: | ---: |
| 按已有checkpoint自动填写 | 完整Validation100的实测分数 |

请等该方法结束再回传最终图片。若某点失败，其他点仍会继续；表格显示“—”，
同时附上对应报错截图即可。完整checkpoint列表不存在时也请回传status.json里的排除说明截图。
所有OUT目录请保留，便于断点续跑和以后核对。

若四个结果目录在同一机器，可在任一仓库执行以下命令，直接回传合并后的四方法图片：

```bash
python training_curve.py --merge /shared/uv_curve/mgn/status.json \
  /shared/uv_curve/eagle/status.json /shared/uv_curve/aroma/status.json \
  /shared/uv_curve/text2pde/status.json --output-dir /shared/uv_curve/merged
```

合并行取各方法update×4的并集，缺少对应checkpoint时留空、不插值。
无需上传checkpoint、数据或流场预测。横轴统一乘4是本次约定的展示坐标，表格只报告UV相对RMSE。

---

## GLaDiT入口（由我执行，学长无需运行）

仅评测已有预测模型，不重训、不更改原选优记录、不访问Test。使用原CUDA推理环境。
固定EMA0.9999、DDIM 6步、标签0–7八次独立采样，在物理UVP空间float64求均值后评分。行坐标为单卡原update。
保持原预测与物理UV相对RMSE定义。

## 运行

先将RUN设为**一个训练run**的绝对目录；自动发现其checkpoints目录中的.pt/.ckpt/.pth；
没有checkpoints目录时扫描RUN本身，不递归搜索其他run。OUT使用新的独立目录。
其他变量沿用原评测配置及prepared数据，AE必须与动力学checkpoint绑定。
Text2PDE的数据位置沿用LDM YAML；RUN只指向LDM run，AE不参与曲线。
本脚本按checkpoint内部真实update排序，重复副本去重；同update不同checkpoint需显式选择。

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python training_curve.py --run-dir "$RUN" --artifacts "$ARTIFACTS" \
  --data-dir "$DATA_DIR" --output-dir "$OUT"
```

可加 `--checkpoints /absolute/run/checkpoints/a.pt /absolute/run/checkpoints/b.pt` 指定子集；
所有文件必须属于RUN。重复原命令会复用完整成功点，失败/中断点加 `--retry-failed` 重试。
单个checkpoint在独立进程中执行，失败保留日志并继续其他点。
`collector.lock` 防止重复启动；异常强制退出后，确认旧进程已结束才移除残留锁。

## 回传图片

**回传OUT/pages.json列出的全部table_XX.png**。每页最多15行，只有训练投入和UV相对RMSE两列。
每完成一个checkpoint就更新图片；“—”代表缺失、失败或尚未完成，具体原因见status.json。
保留OUT用于续跑；精确分数在status.json，CSV和PNG展示6位有效数字。脚本不保存大体积流场预测，
不汇总GPU时间或其他物理指标；底层原评价器计算后仅提取UV分数。

## 合并四个方法

四个status.json可在任一仓库执行以下命令，无需加载模型：

```bash
python training_curve.py --merge /results/mgn/status.json /results/eagle/status.json \
  /results/aroma/status.json /results/text2pde/status.json --output-dir /results/merged
```

输出按各方法update×4的并集排列，缺点留空、不插值；列顺序MGN、EAGLE、AROMA、Text2PDE。
可再加入GLaDiT的status.json，追加第五列，GLaDiT坐标不乘4。只收到图片时按显示精度转录，
不补造隐藏小数。每种方法仅合并一个run；不同run分别制作表格。

首次交付进行静态检查。正式GPU评价及失败状态以运行输出为准。
