# 论文流场渲染交接

先从已完成的 CylinderFlow Validation 预测生成图片。固定论文现有案例
1095、1013、1086，各方法使用同一轨迹和采样标签 0。保留各自原有
representative samples 中的困难与最差案例及图片，随本次结果一并回传。

## 图片清单

| 内容 | 固定设置 |
| --- | --- |
| 案例 | 1095（正文）、1013、1086（附录） |
| 静态图 | 未来帧 16、32、48、64；对应物理时间 1.28、2.56、3.84、5.12 秒 |
| 动图 | 首帧及未来 64 帧，共 65 帧，12.5 fps |
| 物理量 | 速度模长、逐帧面积加权去均值的压力、三角形线性场导数计算的涡量 |
| 对照 | GT、各方法预测、各方法误差；速度行的误差是 UV 向量差的模长 |
| 色标 | 每条轨迹内所有方法和全部时刻共享；误差单独共享色标，采用物理单位线性色标 |
| 文件 | 每时刻 PNG/PDF，以及 GIF、MP4、色标和来源记录 |

本轮为同一采样标签的单次预测展示。已有论文均值场可视化具有自己的采样口径，
合并排版时标明两者。所有预测保持原边界处理和物理单位；渲染计算不重新归一化。

## 准备预测

优先使用已经完成的 Validation 归档。MGN、EAGLE、AROMA 的正式四卡训练
及独立评价均保存共同 NPZ。Text2PDE 四卡流程将完整结果保存在
`evaluation/validation_v1/` 下，其逐样本记录给出 `prediction_file`。
需要的是选定 checkpoint 的预测，权重及完整数据留在原机器。

若 MGN、EAGLE 或 AROMA 尚未导出完整 Validation，在对应方法仓库、原有
环境和已分配 GPU 上，使用各方法仓库 `CYLINDERFLOW.md` 中的既有评价入口。
以下命令的变量填入该方法本次训练使用的数据、配置和选定权重路径：

```bash
python -m cylinderflow evaluate --dataset "$DATA" --manifest "$MANIFEST" \
  --prepared "$PREPARED" --checkpoint "$SELECTED_CHECKPOINT" \
  --mode validation --device cuda --output-dir "$NEW_VALIDATION_DIR"
```

AROMA 还需追加 `--ae-checkpoint "$SELECTED_AE"`，使用该动力学模型训练时
冻结的 AE。模型若使用自定义配置，追加原 `--config "$CONFIG"`。
MGN/EAGLE 的四卡 checkpoint 可由此既有独立评价入口加载。其输出通常位于
`candidate_000/predictions/trajectory_1095_seed_0.npz`。

Text2PDE 完整四卡流程已包含 Validation 导出；缺失时在原四卡环境使用
其正式 LDM 配置和已选择的 AE/LDM checkpoint：

```bash
torchrun --standalone --nproc_per_node=4 -m tools.cylinderflow_stride8.evaluate_four_gpu \
  --stage ldm --mode validation --config "$LDM_CONFIG" \
  --ae-checkpoint "$SELECTED_AE" --checkpoint "$SELECTED_LDM" \
  --output-dir "$NEW_VALIDATION_DIR"
```

保留原配方和采样协议，导出全过程限定 Validation。GLaDiT 预测由本仓库负责人提供。
如某方法的样例归档缺失或非有限，回传该失败记录，并先交付其余已有结果。

## 一次生成三条案例

在本仓库原有环境执行。每个参数是一个标签与一个带 `{trajectory}` 的文件模板，
模板中的采样标签保持 0。按实际归档位置填写模板，命令会核验轨迹、物理单位、
时间、mesh 和 target 一致性。可先只传已有方法，之后在新输出目录补齐全方法。

```bash
python render_comparison.py \
  --method MGN '/shared/mgn/validation/candidate_000/predictions/trajectory_{trajectory}_seed_0.npz' \
  --method EAGLE '/shared/eagle/validation/candidate_000/predictions/trajectory_{trajectory}_seed_0.npz' \
  --method AROMA '/shared/aroma/validation/candidate_000/predictions/trajectory_{trajectory}_seed_0.npz' \
  --method Text2PDE '/shared/text2pde/selected_samples/trajectory_{trajectory}_seed_0.npz' \
  --output-dir reports/paper_flows
```

Text2PDE 的 `selected_samples` 为示例目录，实际位置从逐样本记录中的
`prediction_file` 获取。GLaDiT 的归档文件名通常为 `trajectory_{trajectory}_seed0.npz`，
可通过追加 `--method GLaDiT '实际模板'` 一同绘图。
现有困难案例可追加 `--cases 1095 1013 1086 其他Validation编号`，保持原选择记录。

## 回传

回传输出目录内三条案例的 PNG/PDF、GIF/MP4、`scales.json`、`sources.json`、
`render.json` 及根目录 `handoff.json`。同时提供 checkpoint 的选取依据和已有
困难案例图片。共享前按机器的数据传输约定处理来源记录中的内部路径。
保留服务器原始 NPZ，方便后续统一排版和补图。`handoff.json` 的完成标记在全部
案例渲染成功后写出；异常退出时保留终端错误和已生成内容，使用新目录重跑。

本入口沿用已有渲染依赖。图像布局与静态检查的本地核验和目标机器完整渲染结果
分别记录在交付消息中。Airfoil 案例随其实际结果确定，保持其 0.0016 秒时间间隔。
