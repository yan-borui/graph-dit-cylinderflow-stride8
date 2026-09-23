# NAS 实验入口

六个仓库、17个分支提供统一的 NAS 包装入口。进入所需分支的独立代码目录，激活原有环境，在原启动命令前添加 `bash scripts/nas.sh`。各链接给出完整路径设置与运行命令。

## 各方法入口

| 方法 | CylinderFlow | Airfoil |
| --- | --- | --- |
| VGAE | [运行说明](https://github.com/yan-borui/vgae-cylinderflow-stride8/blob/main/NAS.md) | [运行说明](https://github.com/yan-borui/vgae-cylinderflow-stride8/blob/feature/airfoil-uvp-4gpu/NAS.md) |
| DiT | [运行说明](https://github.com/yan-borui/graph-dit-cylinderflow-stride8/blob/main/NAS.md) | [运行说明与注意力消融](https://github.com/yan-borui/graph-dit-cylinderflow-stride8/blob/feature/airfoil-uvp-4gpu/NAS.md) |
| MGN | [运行说明](https://github.com/yan-borui/meshgraphnets-cylinderflow-stride8/blob/main/NAS.md) | [运行说明](https://github.com/yan-borui/meshgraphnets-cylinderflow-stride8/blob/feature/airfoil-uvp-4gpu/NAS.md) |
| EAGLE | [运行说明](https://github.com/yan-borui/eagle-cylinderflow-stride8/blob/main/NAS.md) | [运行说明](https://github.com/yan-borui/eagle-cylinderflow-stride8/blob/feature/airfoil-uvp-4gpu/NAS.md) |
| AROMA | [运行说明](https://github.com/yan-borui/aroma-cylinderflow-stride8/blob/main/NAS.md) | [运行说明](https://github.com/yan-borui/aroma-cylinderflow-stride8/blob/feature/airfoil-uvp-4gpu/NAS.md) |
| Text2PDE | [运行说明](https://github.com/yan-borui/text2pde-multigeometry-1plus64/blob/main/NAS.md) | [运行说明](https://github.com/yan-borui/text2pde-multigeometry-1plus64/blob/feature/airfoil-uvp-4gpu/NAS.md) |

## 补充实验

- 重复采样：[AROMA](https://github.com/yan-borui/aroma-cylinderflow-stride8/blob/feature/sampling-ensemble/NAS.md)、[Text2PDE](https://github.com/yan-borui/text2pde-multigeometry-1plus64/blob/feature/sampling-ensemble/NAS.md)。
- [MGN 前65帧训练](https://github.com/yan-borui/meshgraphnets-cylinderflow-stride8/blob/feature/train-prefix65/NAS.md)。
- [CylinderFlow 四卡注意力消融](https://github.com/yan-borui/graph-dit-cylinderflow-stride8/blob/feature/attention-ablation-4gpu/NAS.md)。
- [L20 32卡 Scaling](https://github.com/yan-borui/graph-dit-cylinderflow-stride8/blob/feature/scaling-32gpu/NAS.md)。
- [冻结源码上的新训练种子](campaigns/training_seed/README.md)。

## 使用要点

共享准备和运行目录使用原子目录锁，适配缺少 `flock` 命令或文件锁不可用的 NAS。新入口在 Python 启动前设置 HDF5 文件锁兼容模式，具体设置见各分支说明。权重、数据、模型、GPU数量、训练与评价协议沿用各自实验约定。

已有任务继续使用自己的冻结源码。修复版创建新任务时使用独立结果目录；新任务随后可按原恢复入口继续。共享准备的所有参与进程应使用同一版锁协议。异常退出留下的锁目录记录持有者，确认进程及其子进程已退出后再处理。

交付已通过 Python/Bash 静态语法、命令、链接和 Git diff 检查。正式 NAS/GPU 运行由目标环境的原入口验证，Test 继续封存。
