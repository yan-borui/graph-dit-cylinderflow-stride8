# 滑动窗口续生成

使用已有 CylinderFlow 论文模型和冻结 VGAE，输入真实第 0 帧，生成更长的轨迹。默认保持 S6/K8、FP32、单张 CUDA GPU；64 帧窗口每次前进 32 帧，生成未来 256 帧（20.48 秒）。这是一项固定权重的推理实验。

## 窗口与状态

窗口依次以全局帧 0、32、64、96、128、160、192 为条件，分别预测之后的 64 帧。第 32 帧来自首个窗口；第 64 帧来自前两个窗口在该帧的物理 UVP 均值。每次都将当前融合状态重新送入原 VGAE 编码器，构建该窗口的条件。

同一窗口的 K 个成员先按原评测器的 FP32 算术取物理场均值。不同窗口对同一全局帧的预测再等权平均；条件帧不作为额外预测参与平均。因此第二个窗口实际融合帧 33–64。算法仅修改当前窗口起点之后的帧；下一次调用前，起点及之前的输出已经确定。首帧和入口/壁面 UV 继续使用原始条件，压力保留预测值。

未来真值在整条轨迹生成完成后才读取。标准评测只使用原任务的前 65 帧，分别保存首个完整窗口和滑动融合结果在未来 1–64 帧上的原生物理指标。后者增加了重叠窗口计算，两个分数不构成等成本对照。第 65 帧起的生成只供观察长时行为，没有真值误差指标。

## 运行

在本仓库根目录、所选权重对应的既有 CUDA 环境中运行。先确认生产环境和资源；路径指向原训练数据、匹配 artifacts 及固定的选中权重。输出必须为新目录。

```bash
python -m graph_dit.sliding_window \
  --run /absolute/path/to/selected_run \
  --data-dir /absolute/path/to/original_training_data \
  --artifacts /absolute/path/to/matching_artifacts \
  --scope validation --case-count 3 \
  --future-frames 256 --stride 32 \
  --sampling-steps 6 --ensemble-size 8 \
  --device cuda:0 --output-dir /absolute/path/to/new_sliding_result
```

`--run` 读取既有 `selection.json`，不重新选权重；也可替换为 `--checkpoint /.../dit_ema.pt --weights ema_0.9999`。表示和 checkpoint 通过原 `load_selected` 校验。默认取 split 顺序中的前 3 条轨迹；可用 `--indices 1000 1001 1002` 明确指定 Validation 索引。保留所有指定样本及失败记录，固定 `--sampling-label 0`；不筛选随机种子。

Test 使用 [TEST.md](TEST.md) 的官方 Test65 转换目录，将 scope 改为 `--scope test --test-data-dir /absolute/path/to/test65`。默认取 Test 0、1、2；原 Train 数据目录仍用于表示和归一化绑定。此入口不直接接受旧 `test75.h5`，也不改变既有 Test100 评测协议。

每个窗口、每个成员的种子固定为 `((member+1)*1000003 + index*9176 + start*104729 + label*170003) % (2**31-1)`。第一个窗口 label0 与原论文评测种子一致；后续窗口用独立噪声。

## 结果

- `identity.json`、`request.json`、`source.zip`：权重与表示身份、配置、数据划分、实际 CUDA 环境及推理源码。
- `trajectory_*/window_*.npz`、`windows.json`：每个完整窗口的物理场均值、输入状态、种子、帧号和耗时；`--save-members` 额外保存每个成员。
- `trajectory_*/rollout.npz`：完整融合轨迹、参考前缀、网格、每帧贡献次数和物理时间。
- `summary.json`：指定样本的完成/失败记录、前 64 帧的两个指标集合，以及全时段相邻帧 UV 变化幅度。错误记录保留在 `failure.json`；存在失败时入口非零退出。
- `rollout.gif`、`frame_*.png`：速度、压力和涡量的共享色标对照，参考终点后对应面板留空。压力逐帧减去面积加权空间均值；固定色标来自完整参考前缀，超出色标的预测比例保留在 `visual_scales.json`。`--no-render` 仅省略绘图。

默认 256 帧需要 7 次窗口生成，每次 K=8，共 56 次模型采样。窗口结果随生成落盘；推理异常保留已完成窗口和失败记录。

## 验证状态

2026-11-15：Python 3.10 语法、Ruff E9/F 和差异空白检查通过；尚未在匹配的 CUDA 环境中执行推理，暂无生成结果。此模块复用当前 `main` 的原生预测器；部署时保留匹配源码、权重、VGAE 和归一化，并在分配的生产环境中使用此入口运行。
