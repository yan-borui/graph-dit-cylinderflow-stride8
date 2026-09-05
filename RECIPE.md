# 可执行训练 recipe

## 模型与输入

物理 UVP 在原始三角网格上，经固定的 Train-only affine normalizer 转成无量纲训练张量。VGAE 使用 `[2,2,1]` 层级深度、width126、latent channel1；posterior mean 是确定性的表示，随机后验样本不进入 DiT cache。图嵌入与粗图连边也在缓存中固定。

DiT 张量为 `[B,S,N,C]`：B=1，S=1+64，N 是该轨迹的第三层粗图节点数，C=1。内部 width256/384/512 是 token 特征维数，和 latent channel1 分开。输入包含干净首帧 latent、未来64帧的加噪 absolute latent、静态图条件和时空坐标。训练目标是未来64帧上等权 epsilon MSE，1000 个线性 beta diffusion levels，beta 从1e-4到2e-2；每个样本共享一个随机 diffusion level。

H1 在每个 block 内允许同一粗节点或一跳粗图邻居跨全部65个物理帧的非因果注意力。多层网络和20次迭代采样会重复传播信息。当前实现是 dense additive mask，不能根据一跳连边数量推导稀疏加速。depth8/12 改变单次 denoiser 的层数；物理未来64帧、20个DDIM步和训练optimizer update分别计数。

## 共享表示

默认使用 release `representation-v1` 的 `vgae_stride8_epoch930.pt`。这份已训练表示来自固定 stride-8 流程：seed0、batch16、Adam lr1e-4、global normalized UVP MSE + 1e-6 KL、gradient clip1、每epoch每Train轨迹确定性抽一帧0..74、max5000epochs。ReduceLROnPlateau 观察 Train total，factor0.1、patience50，原生结束条件为lr<1e-8。Validation posterior mean在frame0/25/50/74上评价，每10epochs及epoch1产生候选，以 total loss 选优。原运行结束于1092epochs，best epoch930的Validation total为0.023929299879819156。

这份数字是VGAE的规范化重建/KL选择目标，不能和DiT的共同物理指标直接比较。VGAE输出误差也不是所有生成模型误差的严格数学下界。

`representation prepare` 用同一AE编码全部75,000个Train帧，并从这些Train latent节点/帧计算 mean/std；DiT只读取每轨迹的前65帧。Validation物理场不进入这些统计。预处理会产生明确的representation ID、cache/artifact ID、normalizer、数据revision和阶段身份，训练/恢复/评价均核对依赖。所有候选共享同一个`artifacts/shared`目录。标识符用于可信产物的依赖接线；它们不是密码学完整性校验。

需要从头复现表示时只跑一次：

```bash
python -m graph_dit.ae --data-dir data/stride8 --output-dir runs/ae --device cuda:0
# 中断后：同一命令加 --resume，从上次完成的epoch恢复。
python -m graph_dit.representation prepare --data-dir data/stride8 \
  --autoencoder runs/ae/best.pt --output-dir artifacts/shared_rebuilt --device cuda:0
```

重新训练的AE具有新身份，全部DiT候选需要一致地使用它。默认独立DiT seeds复用一个AE，因此只度量该冻结表示下的动力学训练波动。

## 固定训练细节

- AdamW，betas=(0.9,0.999)，eps=1e-8，weight_decay=1e-6，裁剪前梯度范数写日志，clip=1。
- H1、8 heads、MLP ratio4、microbatch1、accumulation1、effective batch1，固定start0；每epoch包含1000个窗口。
- 每个epoch使用seed与epoch确定的轨迹排列；训练噪声使用独立generator；所有候选使用相同seed时共享排列和扩散噪声序列。
- 默认FP32、TF32关闭。已有Full混合精度梯度诊断支持先把算术精度固定作为调参起点；这不构成H1一定存在同样问题的证据。可在计划生成前修改base为BF16并做目标GPU验收；整轮候选使用同一精度，另立新campaign，不在resume时切换。
- 每50,000updates保留不可变checkpoint；每5,000updates写原子替换的`recovery_latest.pt`，避免仅有末态恢复点。程序启动也保存update0恢复状态。
- 每100updates保存epsilon loss、实际LR、裁剪前梯度范数、物理样本ID、时间和显存。无依据loss、梯度、资源或ETA自动剪枝的功能；预定预算完成或实际数值/系统异常才结束当前作业。

固定VGAE的batch16与本次DiT的batch1属于不同训练阶段。

## 学习率、时长和EMA搜索

第一轮完整矩阵为54个配置，见`configs/search.json`。可在生成计划前缩小或扩展候选集合；实际发布结果必须附原计划及全部失败记录。

令k为即将执行的、从1开始的optimizer update，峰值学习率为eta，floor=1e-6，warmup W=4000，固定schedule endpoint U=1,000,000。前W步使用eta*k/W。warmup之后：

| 日程 | 定义 |
| --- | --- |
| constant | eta保持不变 |
| cosine | floor + (eta-floor)*(1+cos(pi*(k-W)/(U-W)))/2 |
| late_decay | 从k=150,000开始乘0.5，随后每50,000updates再次乘0.5，最低floor |

学习率候选为1e-5、3e-5、1e-4。width候选256/384/512，depth候选8/12。stage endpoint控制这次分配多少计算；schedule endpoint控制从开始就固定的LR曲线。第一段250k自然结束，extend从原始raw权重、Adam状态、EMA、sample cursor和RNG继续到1M；原日程不会被缩放成250k cosine再重启。

每50k checkpoint在统一Validation-24、sampling labels0/1/2上评价raw/EMA0.999/EMA0.9999。相同checkpoint各状态使用相同派生seed。EMA按每次optimizer update更新，初始化为初始raw权重，参数使用`ema=beta*ema+(1-beta)*raw`，buffers直接复制。0.999/0.9999的平滑尺度约1000/10000updates。EMA只增加权重副本与评价开销，三种状态共享同一次训练。

训练时长通过相同LR计划中50k、100k、150k……1M的完整生成质量曲线选择；允许更早的checkpoint胜出。独立种子确认仍执行固定1M预算和固定候选日程，以观察后期退化并保留训练成本。若要比较不同cosine终点，必须从头产生新配置；它同时改变调度和时长，不能伪装成相同训练前缀。

该方案参考[官方DiT训练代码](https://github.com/facebookresearch/DiT/blob/main/train.py)的EMA做法，以及[EDM2对训练动态和EMA的研究](https://arxiv.org/abs/2312.02696)。候选值是本任务的待验证设计，原论文不保证它们在CylinderFlow上最优。

## 选择与预算

Checkpoint先按失败clip数、完整轨迹平均UV relative RMSE、update升序排序；完全相同再以预定weights顺序打破平局。EMA在screen/extend参与同等机会的选择；freeze后固定所选raw/EMA类型，独立训练种子只在这个固定类型的共同checkpoint时点选优。候选的预测、标量结果和失败分母完整留存。

全部54个screen作业预定结束后，`promote`选择6个零失败候选；失败和缺失作业不能被静默删掉。`freeze`只接受extend阶段，输出一个锁定配方和seed101/102/103三个新作业。确认种子不相互排序挑选优胜者。排名是有限搜索内的Validation选择，不保证全局最优；早期低LR或大模型可能学习较慢，应结合完整曲线解释未晋级候选。

默认screen是13.5M更新；6个晋级候选额外4.5M；3个确认种子3M，共21M DiT更新，加共享VGAE/cache准备、Validation、报告和测速。单个完整训练每种权重有20个50k候选，screen每种5个；搜索成本和选优机会都必须与冻结baseline配方分开披露。先用目标设备preflight和第一轮实际日志估算GPU-hours，不将其他显卡的速度直接外推。

每个持久checkpoint包含raw、两套EMA与Adam状态，约为模型参数FP32字节数的5倍，另加buffer/RNG。默认容量的源模型约9.8M参数，单份checkpoint约200MB量级；较大模型按实测参数量增长。每种权重、每个monitor时点保存72份完整NPZ，磁盘也要为预测、参考与pre-boundary数组预留空间。不得覆盖早期checkpoint来节省空间；确需调整保留策略时先形成新的明确交付配置。

精确恢复的CPU验收比较了raw、EMA、Adam、训练generator和sample cursor。CUDA散射和attention kernels可能不满足位级确定性；集群运行记录源码、设备、环境和随机状态，按数值与科学指标验证复现。
