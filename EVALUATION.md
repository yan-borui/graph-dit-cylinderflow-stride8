# 共同评价、媒体和成本

## Checkpoint选择与完整Validation

选择集合与四个baseline一致：Validation1000..1099中`np.rint(linspace(0,99,24))+1000`的24条轨迹。每个完整64帧预测分别评分，先在轨迹内平均sampling labels0/1/2，再对轨迹等权平均；任何一个clip失败，该轨迹不产生有限均值，同时保留失败clip数、失败trajectory数和完整分母。先按失败数、再按平均UV relative RMSE、再按较早update排序。

训练每50k时点比较raw与两套EMA，所有已评分预测都存成共同NPZ。`selection.json`指出具体checkpoint和EMA类型。最终冻结后，每个独立training seed在同一固定EMA类型上选checkpoint，再运行：

```bash
RUN=campaigns/fp32_confirm/runs/confirmation_seed101
python -m graph_dit.evaluate --run "$RUN" \
  --data-dir "$DATA_DIR" --artifacts "$ARTIFACTS" --device cuda:0 \
  --scope validation --output-dir runs/validation_seed101
```

对seed102/103重复执行并报告全部结果。每个run的完整Validation是100条轨迹×3条sampling seeds=300clips；这是100条独立轨迹，不能按300个独立样本计算置信区间。完整Validation用于报告冻结方案；继续使用这些结果开发新方案时要保留其已使用身份。Test仍需要单独授权，本仓库没有Test入口。

## 物理口径与数组

共同`metrics.py`、`predictions.py`、`performance.py`原样复用MGN发布提交7ef6fff的实现，也对应另外三个baseline。预测及target都在物理UVP单位中只反标准化一次。

- UV relative RMSE：对future1..64、速度两分量和节点求triangle-derived面积加权平方误差与参考能量之比，再开方。
- pressure同时保留原始和逐帧面积加权去常数偏移的误差；vorticity/divergence使用三角形上线性场导数。
- 保留energy、enstrophy、时间频谱、相位、逐帧与末帧误差、P90和边界指标。边界子集误差采用单独声明的无面积节点平均。
- 仅inlet/wall的UV写回首帧值；pressure和outlet自由预测。Graph DiT先联合采样、解码，再写回；不在latent采样中反复用未来真值修正。

NPZ包含`prediction`、`pre_boundary`、`target`，均为物理`[65,N,3]`，以及points/cells/node_type、trajectory_index、seed、raw indices `0,8,...,512`、time `0,0.08,...,5.12`、units和JSON provenance。输入到`Predictor.predict`的对象只有真实首帧和静态网格；未来reference在推理完成后才读取。数值失败也会存NaN占位和完整首帧，不能静默删除。

## 补评分和配对视频

```bash
python -m graph_dit.review score --inputs runs/validation_seed101/predictions/*.npz \
  --output-dir reports/validation_seed101_rescored

python -m graph_dit.review render \
  --inputs runs/validation_seed101/predictions/trajectory_1000_seed0.npz \
           /shared/mgn/predictions/trajectory_1000_seed0.npz \
  --labels Graph_DiT MGN --output-dir reports/paired_1000
```

配对输入必须是相同mesh、target、时间和轨迹。renderer同时画GT、各方法和误差，生成速度、gauge-free pressure、vorticity的GIF/MP4与静态图；同组案例/方法/整个时间使用固定共享色标。sampling label固定0，不能从三条采样中挑最好的一条。`report --run ... --movies`从已选monitor结果按轨迹得分透明选median/P90/worst，并保留所有原始不利案例；完整Validation可用上面的render命令另画共同案例。

`report --run`生成epsilon loss、LR、裁剪前gradient norm、raw/EMA物理Validation曲线和运行卡。`report --plan`生成全候选精确数值页，包含未开始、运行、失败和完成状态。EMA/raw的评价开销及训练更新耗时分项留在status/checkpoint中，训练总占卡时间另计。

## 独立共同测速

```bash
python -m graph_dit.benchmark --run "$RUN" \
  --data-dir "$DATA_DIR" --artifacts "$ARTIFACTS" \
  --device cuda:0 --output-dir runs/performance_seed101
python -m graph_dit.performance \
  --reports runs/performance_seed101/summary.json /shared/mgn/performance/summary.json \
            /shared/eagle/performance/summary.json /shared/aroma/performance/summary.json \
            /shared/text2pde/performance/summary.json \
  --output reports/matched_cost.json
```

计时范围为CPU中已就绪的物理首帧与静态mesh缓存，到完整CPU物理UVP输出；包含标准化、传输、初始编码、完整20-step DDIM、64帧解码和边界写回。静态coarsening/hop缓存构造、模型/文件加载、warmup、评分和渲染分别记录或排除。实际采样随机种子在计时外统一设置，且同卡FP32、无TF32/autocast，每条Validation-24预热2次、测3次。

不能用原生quality-evaluator的`inference_seconds`代替共同测速结果。模型参数量与存储包括VGAE；训练、表示准备、缓存、Validation成本分别披露。共同报告合并会检查设备、软件环境、registry、完整性和失败；实际GPU独占状态由目标调度分配记录证明。协议全文见[PERFORMANCE.md](PERFORMANCE.md)。

## 交接材料

对方保留模型、normalizer、缓存、完整预测、配置、源码、日志与失败证据。回传内容按双方约定：PNG配置卡和数值页、训练/Validation曲线、共享色标GIF/MP4、成本及失败页；允许时附精简JSON/CSV。报告图片正常呈现数值，不把原始权重或大文件编码成媒体绕过传输限制。已有预测支持后续加指标或重画图，双方应保留相应存储和补跑入口。
