# CylinderFlow Test 评测

本分支支持官方 Test100 的下载、独立转换和评测。使用已有环境，从仓库根目录执行。将下面的路径变量替换为对应已训练运行的路径，`OUT` 使用全新结果目录。

## 各方法入口

各方法使用自身原生推理和评分方式，具体参数、依赖和输出见对应分支说明。

| 方法 | CylinderFlow | Airfoil |
| --- | --- | --- |
| GLaDiT | 本页 | [Airfoil](https://github.com/yan-borui/graph-dit-cylinderflow-stride8/blob/feature/airfoil-uvp-4gpu/TEST.md) |
| MGN | [前 65 帧训练](https://github.com/yan-borui/meshgraphnets-cylinderflow-stride8/blob/feature/train-prefix65/TEST.md) / [75 帧训练](https://github.com/yan-borui/meshgraphnets-cylinderflow-stride8/blob/cylinderflow-stride8/TEST.md) | [Airfoil](https://github.com/yan-borui/meshgraphnets-cylinderflow-stride8/blob/feature/airfoil-uvp-4gpu/TEST.md) |
| EAGLE | [prefix65 分支](https://github.com/yan-borui/eagle-cylinderflow-stride8/blob/feature/train-prefix65/TEST.md) / [stride8 分支](https://github.com/yan-borui/eagle-cylinderflow-stride8/blob/cylinderflow-stride8/TEST.md) | [Airfoil](https://github.com/yan-borui/eagle-cylinderflow-stride8/blob/feature/airfoil-uvp-4gpu/TEST.md) |
| AROMA | [prefix65 分支](https://github.com/yan-borui/aroma-cylinderflow-stride8/blob/feature/train-prefix65/TEST.md) / [stride8 分支](https://github.com/yan-borui/aroma-cylinderflow-stride8/blob/cylinderflow-stride8/TEST.md) | [Airfoil](https://github.com/yan-borui/aroma-cylinderflow-stride8/blob/feature/airfoil-uvp-4gpu/TEST.md) |
| Text2PDE | [1+64 分支](https://github.com/yan-borui/text2pde-multigeometry-1plus64/blob/feature/cylinderflow-stride8-1plus64/TEST.md) | [Airfoil](https://github.com/yan-borui/text2pde-multigeometry-1plus64/blob/feature/airfoil-uvp-4gpu/TEST.md) |
| VGAE 重建 | [CylinderFlow](https://github.com/yan-borui/vgae-cylinderflow-stride8/blob/main/TEST.md) | [Airfoil](https://github.com/yan-borui/vgae-cylinderflow-stride8/blob/feature/airfoil-uvp-4gpu/TEST.md) |

三个消融均支持两个数据集：[确定性图 Transformer](https://github.com/yan-borui/graph-dit-cylinderflow-stride8/blob/feature/deterministic-graph-transformer-4gpu/TEST.md)、[确定性 GNN U-Net](https://github.com/yan-borui/graph-dit-cylinderflow-stride8/blob/feature/deterministic-joint-4gpu/TEST.md)、[图潜空间扩散](https://github.com/yan-borui/graph-dit-cylinderflow-stride8/blob/feature/graph-latent-diffusion-4gpu/TEST.md)。

## 下载与转换

下载脚本只依赖 Python 标准库，支持断点续传和共享目录锁。两个数据集原始 Test 合计约 6.41 GB；下载不需要模型权重。

```bash
export RAW_TEST_ROOT=/shared/data/mgn-test-raw
bash scripts/download_test.sh --dataset both --raw-root "$RAW_TEST_ROOT"
```

官方来源：[Airfoil Test](https://storage.googleapis.com/dm-meshgraphnets/airfoil/test.tfrecord)、[CylinderFlow Test](https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord)，以及对应目录的 `meta.json`；与 [MeshGraphNets 官方下载入口](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/download_dataset.sh) 一致。

本分支评测 **CylinderFlow**。原始 Test 文件可以跨方法共享；转换目录绑定训练 manifest 的归一化，统计不同的运行使用各自目录。

```bash
export TRAIN_MANIFEST=/absolute/path/to/original_training_manifest.json
export TEST_DIR=/shared/data/cylinderflow-test/selected-normalization/test65
python -m mgn_data.prepare_test \
  --dataset cylinderflow --raw-dir "$RAW_TEST_ROOT/cylinder_flow" \
  --train-manifest "$TRAIN_MANIFEST" --output-dir "$TEST_DIR"
```

转换仅保留 raw 0, 8, …, 512，共 65 帧 UVP。Test 保持独立的 100 条官方轨迹，manifest 的 split 为 `test`。归一化逐项复制对应 Train manifest；已有相同完整转换可复用，失败转换保留在独立临时目录。转换检查和转换所需依赖来自已有 NumPy/h5py 环境。

## 使用选定权重

```bash
export OUT=/absolute/path/to/new_test100_result
python -m graph_dit.evaluate \
  --run "$RUN" --artifacts "$ARTIFACTS" --data-dir "$TRAIN_DATA_DIR" \
  --scope test --test-data-dir "$TEST_DIR" \
  --sampling-steps 6 --ensemble-size 8 \
  --device cuda:0 --output-dir "$OUT"
```

`RUN` 包含原有 `selection.json` 和其指向的权重；`ARTIFACTS` 为同一表示的 VGAE 导出及 latent 统计，`TRAIN_DATA_DIR` 保留原训练数据绑定。上述命令使用 S6/K8：每成员 6 步去噪，8 个完整成员解码后取物理 UVP 均值，再对每条轨迹评分。 本分支不指定采样参数时仍沿用旧入口的 S20、3 次独立采样评分；复现论文 S6/K8 时保留上面的两个显式参数。

输出含 `identity.json`、`summary.json`、逐样本和逐轨迹 CSV、`failures.json` 及预测文件。记录明确包含 Test 身份、训练绑定、权重、S/K 和聚合方式。

## 比较口径与验证范围

CylinderFlow 的物理帧间隔为 0.08，条件为帧 0，预测和评分覆盖后续 64 帧。沿用模型的原生边界处理、推理和指标。Train 统计、原权重选择记录以及已经报告的 Validation 结果分别保留；补测结果标明 Test、采样预算、聚合方式和失败数。历史权重的 Train75 或前 65 帧训练暴露继续按原运行记录报告。

本次入口交付仅完成静态语法、引用和 diff 检查。正式 GPU 推理及完整 Test 转换尚待目标环境执行；本次代码更新没有产生新的 Test 指标。
