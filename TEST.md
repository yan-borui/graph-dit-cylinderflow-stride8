# 确定性图 Transformer Test 评测

本分支沿用已有 Test 转换器和评测入口，另提供独立下载命令。从仓库根目录使用已有环境执行。

## 下载

```bash
bash scripts/download_test.sh --dataset both --raw-root /shared/data/mgn-test-raw
```

该命令只依赖 Python 标准库，下载两个数据集各自的 `meta.json` 与 `test.tfrecord`，合计约 6.41 GB。原始文件可供全部方法共享；下载支持中断续传，并与转换入口共用原始目录锁。

官方来源：[Airfoil Test](https://storage.googleapis.com/dm-meshgraphnets/airfoil/test.tfrecord)、[CylinderFlow Test](https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord)，与 [MeshGraphNets 下载脚本](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/download_dataset.sh) 一致。

## 转换与完整评测

沿用对应数据集训练时的 `WORK_DIR`、`CACHE_DIR`、`RESULT_ROOT`、投稿表示及身份设置；`OUTPUT_DIR` 必须是新目录。下面两条命令分别在各自运行的环境变量设置下执行。

```bash
RAW_DIR=/shared/data/mgn-test-raw/airfoil \
  TEST_DATA_DIR=/shared/data/airfoil-ablation-test SPLIT=test OFFLINE=1 \
  OUTPUT_DIR=/shared/results/airfoil-test100 \
  bash scripts/deterministic_4gpu.sh airfoil eval

RAW_DIR=/shared/data/mgn-test-raw/cylinder_flow \
  TEST_DATA_DIR=/shared/data/cylinderflow-ablation-test SPLIT=test OFFLINE=1 \
  OUTPUT_DIR=/shared/results/cylinderflow-test100 \
  bash scripts/deterministic_4gpu.sh cylinderflow eval
```

首次评测自动在 `TEST_DATA_DIR/test65/` 发布完整 HDF5 和 manifest，仅保留 raw 0, 8, …, 512 的 UVP，使用所选 VGAE 绑定的 Train 统计。已完成且身份匹配的转换可复用；不同归一化使用不同数据根。

评测从原 `selection.json` 读取已选权重，覆盖官方 Test100，首帧条件预测后续 64 帧。一次确定性前向。Airfoil 的物理步长为 0.0016，预测全部节点；CylinderFlow 为 0.08，沿用首帧已知入口和壁面速度。失败保留在全量记录中，汇总报告失败数并以非零退出码标记不完整结果。

输出保留逐轨迹指标、物理场和 `summary.json`。Test 与已有 Validation 结果按 split 分别报告，原权重选择记录保持一致。静态代码检查已经完成；本次更新未执行新的 GPU 评测，也未产生新的 Test 指标。训练、表示绑定及采样参数详见[DETERMINISTIC.md](DETERMINISTIC.md)。
