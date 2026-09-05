# 环境与安装

建议Linux、Python3.11与目标GPU驱动兼容的PyTorch/torchvision配套环境。先选目标集群支持的PyTorch构建，再安装本仓库；优先复用可用环境，不改系统Python。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 依目标驱动选择匹配的官方CUDA wheel或集群提供的PyTorch容器：
# https://pytorch.org/get-started/locally/
python -m pip install -e .
python prepare.py --device cuda:0
```

`pyproject.toml`声明必要依赖；公开版本还保存`environment-observed.txt`作为已验证运行时的版本记录。它记录了一次具体环境，不代表所有GPU都应安装相同CUDA构建。不要单独升级torchvision而保留不兼容的PyTorch。

共同公开HDF5由HDF5 2.0写出，所用h5py必须链接能够读取该文件的HDF5运行时。先检查`python -c 'import h5py; print(h5py.version.info)'`并实际运行下载器的数据合同检查；本次观察到h5py3.16.0可读。若集群既有HDF5过旧，可使用新的隔离环境或经数据发布者认可的转换副本，并保留新文件的身份及逐数组一致性记录。不要把文件读取错误当成模型问题。

模型使用PyG的图对象、coalesce和sum scatter，以及网格给定连边；这条路径不调用kNN搜索，安装无需额外编译torch-cluster/torch-scatter。它们是PyG的可选扩展；若既有环境装了不匹配的扩展，应在自己的隔离环境中处理兼容性。

本轮默认训练使用FP32前向和反向，显式关闭AMP和TF32。启动记录保存实际参数dtype、矩阵乘法精度和TF32开关；BF16解析供旧配置使用。GPU数量、总显存、实际峰值和每update时间从目标设备preflight取得。宽度512/depth12的最大图验收不能用CPU合成smoke替代。

`prepare.py`复用既有下载器和表示准备流程，不添加依赖。默认从固定版本的Hugging Face仓库下载Train/Validation与VGAE；`--download-only`适用于联网准备节点，`--ae-source github`切换到原有VGAE release。后续执行会复用已完成的资源。

在Windows上可以生成计划、编辑配置和阅读报告；本轮软件训练验收使用Linux CPU。Slurm只用于支持它的集群。没有在目标集群实际提交作业，也没有假定具体partition/account/作业时限。
