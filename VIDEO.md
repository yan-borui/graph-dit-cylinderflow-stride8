# 补充材料视频导出

从已保存的 Validation 物理场生成三组对比视频：Airfoil 方法对比、CylinderFlow 方法对比，以及 CylinderFlow 的 H1/H2/Full 注意力消融。每组输出 MP4 和循环 GIF，包含 GT、预测和误差。脚本直接读取预测文件，在 CPU 上完成选样、集成和渲染。

## 准备输入

在本分支的独立 checkout 中，使用已有的 NumPy、Matplotlib、ImageIO、imageio-ffmpeg 和 Pillow 环境。MP4 编码使用 libx264；已有 FFmpeg 可通过 `IMAGEIO_FFMPEG_EXE` 指定。导出入口的依赖沿用项目清单。

编辑三个配置中的预测位置：

- [Airfoil](configs/video/airfoil.json)：GLaDiT、MGN、EAGLE、AROMA、Text2PDE。
- [CylinderFlow](configs/video/cylinderflow.json)：同样五种方法。
- [注意力消融](configs/video/attention.json)：训练种子 0 的 H1、H2、Full。

示例中的 `/shared/video/` 表示实验端现有归档位置，请按实际目录替换。文件模板保留 `{trajectory}`，相对路径按配置文件所在目录解析。输入绑定各方法已正式选择的检查点，保留原始归档中的 checkpoint、权重、表示和采样信息。注意力配置同时核验归档中的训练种子与 attention 类型。

两种数据集均采用 Validation 编号 1000–1099，数组为 `[65,N,3]`，三个通道依次是物理单位的 U、V、P。65 帧对应原始帧 0、8、…、512。CylinderFlow 的帧间隔为 0.08 秒，Airfoil 为 0.0016 秒。脚本核验同轨迹各方法的节点顺序、网格、节点标签、真值与时间完全一致，并保留预测原有边界值。

支持的归档格式为 `cylinderflow.physical_prediction.v2` 与 `airfoil.uvp.physical_prediction.v1`。来源中应包含检查点身份。格式不匹配时，根据失败记录核对实验端现有导出文件及来源。

## 选样与采样口径

默认扫描 GLaDiT 的完整 Validation100，以 K8 物理均值场在未来 64 帧上的面积加权 UV 相对 RMSE 选择最佳轨迹。分数相同时取较小编号。评分与共同评价器的 UV 定义一致：

$$
\sqrt{\frac{\sum_{t=1}^{64}\sum_i a_i\lVert\hat{\mathbf{u}}_{t,i}-\mathbf{u}_{t,i}\rVert_2^2}{\sum_{t=1}^{64}\sum_i a_i\lVert\mathbf{u}_{t,i}\rVert_2^2}}
$$

其中 $a_i$ 是邻接三角形面积各取三分之一后相加的节点权重。选样使用固定检查点、固定推理配置和预先指定的采样组。候选缺失、非有限或身份不一致时，保存逐案例失败信息并退出，补齐后使用新目录重跑。

GLaDiT 展示 K8 均值场；baseline 固定采样标签 0。视频标题显示数据集和轨迹序号，方法标签注明采样口径。选择依据保存在选样记录中。注意力视频使用主对比已选定的同一轨迹，展示 H1/H2/Full 各自的单次预测，固定训练种子 0 和采样标签 0。

优先填写 `mean_template` 指向既有 `mean_k8.npz`。该文件的来源应记录 `ensemble_size=8`、`aggregation=physical_uvp_mean` 和八个不同的 `member_prng_seeds`。`group` 用于记录固定采样组；请把模板固定到该组的完成目录。

若已保存八个单次成员，可在 GLaDiT 配置中添加以下 `members`。每个 `seed` 与对应 NPZ 的采样标签一致，归档来源中须有实际 `prng_seed` 或 `sample_seed`。脚本沿用逐成员 float64 在线均值、最后转换为 float32 的集成流程。

```json
"members": [
  {"template": "/shared/video/gladit/group0/trajectory_{trajectory}_seed0.npz", "seed": 0},
  {"template": "/shared/video/gladit/group0/trajectory_{trajectory}_seed1.npz", "seed": 1},
  {"template": "/shared/video/gladit/group0/trajectory_{trajectory}_seed2.npz", "seed": 2},
  {"template": "/shared/video/gladit/group0/trajectory_{trajectory}_seed3.npz", "seed": 3},
  {"template": "/shared/video/gladit/group0/trajectory_{trajectory}_seed4.npz", "seed": 4},
  {"template": "/shared/video/gladit/group0/trajectory_{trajectory}_seed5.npz", "seed": 5},
  {"template": "/shared/video/gladit/group0/trajectory_{trajectory}_seed6.npz", "seed": 6},
  {"template": "/shared/video/gladit/group0/trajectory_{trajectory}_seed7.npz", "seed": 7}
]
```

`mean_template` 可与 `members` 同时提供：均值文件存在时校验并采用该文件，缺失时采用完整八成员。来源检查保留每个成员的身份与顺序。

## 三组导出命令

在仓库根目录依次执行，确保每个输出目录为新目录：

```bash
python export_video.py --config configs/video/airfoil.json --output-dir reports/airfoil
python export_video.py --config configs/video/cylinderflow.json --output-dir reports/cylinderflow
python export_video.py --config configs/video/attention.json --output-dir reports/attention
```

注意力配置的 `selection_file` 指向第二条命令生成的选样记录。若 CylinderFlow 使用其他输出目录，同步修改该路径。

可追加 `--trajectory 1018` 沿用当前 Airfoil 轨迹，记录中保留人工指定方式。注意力入口始终沿用主对比的选择。正式影片默认导出全部 65 帧，播放速度为 12.5 fps；可用 `--fps` 修改播放速度，画面上的时间保持真实物理时间。

选样记录在检查 baseline 文件前写入。选中轨迹有缺项时，错误会逐方法列出路径，随后补齐同一案例即可。已成功写出的选样记录也可直接供注意力入口使用。

## 画面与回传

画面按方法分列，六行依次为速度模长、UV 向量误差、去均值压力、压力误差、涡量和涡量误差。压力逐帧去除全域面积加权均值，涡量使用三角形线性场的空间导数。每一物理量的颜色范围覆盖全部方法与全部 65 帧，误差共享独立线性色标。

Airfoil 默认显示 $x\in[-1,3]$、$y\in[-1,1]$，全域网格小图用红框标出显示区域。所有面板共享这一视窗，色标取与视窗相交的三角形及其节点在全部方法、全部帧上的极值。配置中的四个视窗边界可按原始坐标调整；首、中、末帧审核时确认翼型及近尾流完整可见。自定义色标发生截断时，色条端部箭头标明超限方向。全域预测、压力处理和评价数据保持原有定义。

每次导出保存第0、32、64帧预览。先使用同一入口加 `--trajectory 1018 --frames 0 32 64` 审核 Airfoil 的空间范围、色标和清晰度，再用默认全部帧导出到新目录。MP4 使用 H.264 High、YUV420P 和快速起播布局；正式导出后检查目标播放器能从头播放至末帧。

回传以下产物：

| 产物 | 内容 |
| --- | --- |
| `media/comparison.mp4`、`media/comparison.gif` | 对比视频与循环动图 |
| `selection.json` | 候选得分、选样方式、固定采样组及选中案例 |
| `sources.json`、`config.json` | 输入来源、检查点与采样口径 |
| `media/scales.json`、`media/render.json` | 色标、导出帧、物理时间与播放速度 |
| `status.json` | 完成状态；失败时附异常信息 |

输出中的 `inputs/` 保存选中案例的渲染输入，便于复现；原始预测保留在原位置。失败时同时保存 `error.txt`，已生成的文件继续保留。来源记录含机器路径，向外共享时按现有数据约定处理。

## 验证

本次交付执行 Python 语法、导入引用、配置和差异静态检查。实际编码及视觉效果需在正式导出环境核验。确认操作系统、CPU、运行时和相关依赖与正式导出环境一致后，可在同一入口追加 `--frames 0 32 64`，缩减渲染帧数进行一次验收。该参数保留全轨迹选样、物理场计算和共享色标，缩减结果在记录中标明。运行时记录退出码，并检查两种文件可播放、标签与物理时间正确、末帧和误差面板完整。正式导出使用默认全部帧。
