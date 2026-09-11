# 可执行训练 recipe

## 模型与输入

物理 UVP 在原始三角网格上，经固定的 Train-only affine normalizer 转成无量纲训练张量。VGAE 使用 `[2,2,1]` 层级深度、width126、latent channel1；posterior mean 是确定性的表示，随机后验样本不进入 DiT cache。图嵌入与粗图连边也在缓存中固定。

DiT 张量为 `[B,S,N,C]`：B=1，S=1+64，N 是该轨迹的第三层粗图节点数，C=1。内部 width256/384/512 是 token 特征维数，和 latent channel1 分开。输入包含干净首帧 latent、未来64帧的加噪 absolute latent、静态图条件和时空坐标。训练目标是未来64帧上等权 epsilon MSE，1000 个线性 beta diffusion levels，beta 从1e-4到2e-2；每个样本共享一个随机 diffusion level。

H1 在每个 block 内允许同一粗节点或一跳粗图邻居跨全部65个物理帧的非因果注意力。多层网络和20次迭代采样会重复传播信息。当前实现是 dense additive mask，不能根据一跳连边数量推导稀疏加速。depth8/12 改变单次 denoiser 的层数；物理未来64帧、20个DDIM步和训练optimizer update分别计数。

## 共享表示

默认通过`python prepare.py --device cuda:0`下载并复用`vgae_stride8_epoch930.pt`。权重来自[Hugging Face固定revision](https://huggingface.co/DingDong1921/graph-dit-cylinderflow-stride8/tree/8abcd6128a9896d6097d5dccb60cd0beaa4b852e)，也可用`--ae-source github`选择原`representation-v1` release；加载时核对表示ID和数据revision。这份已训练表示来自固定 stride-8 流程：seed0、batch16、Adam lr1e-4、global normalized UVP MSE + 1e-6 KL、gradient clip1、每epoch每Train轨迹确定性抽一帧0..74、max5000epochs。ReduceLROnPlateau 观察 Train total，factor0.1、patience50，原生结束条件为lr<1e-8。Validation posterior mean在frame0/25/50/74上评价，每10epochs及epoch1产生候选，以 total loss 选优。原运行结束于1092epochs，best epoch930的Validation total为0.023929299879819156。

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
- 本轮默认FP32前向和反向，参数、梯度和Adam状态使用FP32，AMP及TF32关闭。FP32入口显式禁用外层autocast，启动记录保存配置精度、实际参数dtype和TF32设置。已有Full混合精度诊断支持在这一精度下比较快慢调度；H1长期稳定性由实际训练结果判断。BF16解析保留用于旧配置；本轮所有候选统一FP32，精度随配置一起冻结。
- 每50,000updates保留不可变checkpoint；每5,000updates写原子替换的`recovery_latest.pt`，避免仅有末态恢复点。程序启动也保存update0恢复状态。
- 每100updates保存epsilon loss、实际LR、裁剪前梯度范数、物理样本ID、时间和显存。无依据loss、梯度、资源或ETA自动剪枝的功能；预定预算完成或实际数值/系统异常才结束当前作业。

固定VGAE的batch16与本次DiT的batch1属于不同训练阶段。

## 学习率、时长和EMA搜索

第一轮完整矩阵为72个配置，见`configs/search.json`：原有54组快降，加18组慢cosine。可在生成计划前缩小或扩展候选集合；实际发布结果必须附原计划及全部失败记录。

本轮同时比较十倍阶梯快降和慢cosine。快速下降保留100k/125k/150k三个起点；慢cosine从warmup结束连续下降至1M。在相同FP32精度、初始种子和模型容量下比较这些候选，观察较长的高LR训练是否有收益。恒定LR保留在配置解析器中，默认搜索使用上述四种日程。

令$k$为即将执行的、从1开始的optimizer update，峰值为$\eta$，下限为$10^{-7}$，warmup为$W=4000$，首次下降时点为$S\in\{100000,125000,150000\}$，后续下降间隔为$P=50000$。完整计划覆盖$U=1000000$次更新。快速下降日程为：

$$
\operatorname{lr}(k)=
\begin{cases}
\eta k/W, & 1\le k\le W,\\
\eta, & W<k<S,\\
\max\left(10^{-7},\eta\,10^{-(1+\lfloor(k-S)/P\rfloor)}\right), & S\le k\le U.
\end{cases}
$$

首次十倍下降发生在第$S$次optimizer更新之前。以$S=150000$为例：

| 峰值LR | warmup结束至149,999 | 150,000至199,999 | 200,000至249,999 | 250,000以后 |
| --- | --- | --- | --- | --- |
| 1e-5 | 1e-5 | 1e-6 | 1e-7 | 1e-7 |
| 3e-5 | 3e-5 | 3e-6 | 3e-7 | 1e-7 |
| 1e-4 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |

100k/125k配置把整组下降时点提前50k/25k。默认base为峰值3e-5、125k首次下降：4k达到3e-5，125k降至3e-6，175k降至3e-7，225k达到1e-7下限。`late_decay`实现读取`decay_start_updates`、`decay_period_updates`、`decay_factor`；本轮固定后两者为50k和0.1，起点在search.json中枚举。任务ID带`drop100000/drop125000/drop150000`，每份生成配置保存确切日程。

慢cosine使用相同warmup；$W<k\le U$时：

$$
\operatorname{lr}_{\mathrm{cosine}}(k)=10^{-7}+\frac{\eta-10^{-7}}{2}\left[1+\cos\left(\pi\frac{k-W}{U-W}\right)\right].
$$

cosine在1M的最终LR为1e-7；它为每组峰值与宽深组合生成一个候选，不重复枚举快降起点。

学习率候选为1e-5、3e-5、1e-4。width候选256/384/512，depth候选8/12。stage endpoint控制这次分配多少计算；schedule endpoint给出预先定义的完整训练范围。第一段250k自然结束，extend从原始raw权重、Adam状态、EMA、sample cursor和RNG继续到1M；续训保持同一个绝对更新日程。

已有运行使用各自原plan及源码继续；本轮FP32、1e-7下限及快慢调度搜索另生成`campaigns/fp32_screen`。更换日程或精度需要新运行，不能在原run中改配置后resume。

每50k checkpoint在统一Validation-24、sampling labels0/1/2上评价raw/EMA0.999/EMA0.9999。相同checkpoint各状态使用相同派生seed。EMA按每次optimizer update更新，初始化为初始raw权重，参数使用`ema=beta*ema+(1-beta)*raw`，buffers直接复制。0.999/0.9999的平滑尺度约1000/10000updates。EMA只增加权重副本与评价开销，三种状态共享同一次训练。

训练时长通过相同LR计划中50k、100k、150k……1M的完整生成质量曲线选择；允许更早的checkpoint胜出。独立种子确认仍执行固定1M预算和固定候选日程，以观察后期退化并保留训练成本。若要比较不同cosine终点，必须从头产生新配置；它同时改变调度和时长，不能伪装成相同训练前缀。

该方案参考[官方DiT训练代码](https://github.com/facebookresearch/DiT/blob/main/train.py)的EMA做法，以及[EDM2对训练动态和EMA的研究](https://arxiv.org/abs/2312.02696)。候选值是本任务的待验证设计，原论文不保证它们在CylinderFlow上最优。

## 2026-09-12 四卡窗口预算搜索

[FOUR_GPU.md](FOUR_GPU.md) 与 `configs/search_4gpu_12_20260911.json` 定义本轮独立配方：
w512-d16/1e-4 以及 w512-d24/1e-4、5e-4、1e-3，交叉 250k/500k/1M 全局窗口 cosine。
每组四卡、global B4，分配 250k 窗口即 62,500 updates；最低 LR **1e-6**，warmup
4k 窗口。EMA0.999/0.9999 按窗口计，实际每更新衰减取四次方。每 50k 窗口评价
Validation-24 × 三个采样 × raw/两套 EMA，关闭早停和自动晋级。原配方继续使用其
既有更新单位、EMA 单位与 LR 下限，历史运行记录不改写。

## 选择与预算

### 4090 单次 cosine + EMA 运行（2026-09-08）

`configs/h1_w512_cosine_ema_4090_20260908a.json` 固定 seed0、H1、width512、depth8、FP32、
峰值1e-4、warmup4000、1M cosine终点与1e-7下限。每5000updates同时保存独立checkpoint和
滚动恢复点，并评价Validation-24 × 三个sampling labels × raw/EMA0.999/EMA0.9999，共216段。
该运行的间隔与终止规则由此独立配置指定。

可选 `validation.early_stopping` 缺省关闭。本配置明确启用 `min_updates=500000`、
`patience_evaluations=20`。每种权重必须72段全部有效才可选优；三种权重的最低有效UV分数
构成一轮结果。主指标严格下降即更新全局最佳并清零等待；持平、变差或三种权重全数值失败
只累计一轮。500k前照常累计，当前update达到500k且连续20轮无改善时正常 `early_stopped`。
提前结束保留原1M学习率日程。结论限于用于选择的Validation-24，Test封存。

评估程序异常保留失败目录与恢复点、打断等待并以故障状态退出；同配置 `--resume` 时按
恢复update重放原子记录，复用已完成权重的评价，补齐当前轮。每轮结束后恢复raw权重、模型
模式与训练随机数。`physical_monitor/state.json` 保存最佳与等待，`selection.json` 指向
最佳checkpoint和raw/EMA类型；终态 `checkpoint_inventory.json` 同时索引最后和物理最佳
checkpoint，`status.json` 区分早停与跑满预算。全部checkpoint、预测及失败证据保留。

新增早停验收复用已有CPU smoke产物：

```bash
python -m graph_dit.early_stop_acceptance --smoke-root runs/cpu_smoke --output-dir runs/early_stop_acceptance
```

以下搜索流程仍使用各自冻结配置及预算。

Checkpoint先按失败clip数、完整轨迹平均UV relative RMSE、update升序排序；完全相同再以预定weights顺序打破平局。EMA在screen/extend参与同等机会的选择；freeze后固定所选raw/EMA类型，独立训练种子只在这个固定类型的共同checkpoint时点选优。候选的预测、标量结果和失败分母完整留存。

全部72个screen作业预定结束后，`promote`选择6个零失败候选；失败和缺失作业不能被静默删掉。`freeze`只接受extend阶段，输出一个锁定配方和seed101/102/103三个新作业。确认种子不相互排序挑选优胜者。排名是有限搜索内的Validation选择，不保证全局最优；早期低LR、慢cosine或大模型可能需要更长训练，应结合完整曲线解释未晋级候选。

默认screen是18M更新；6个晋级候选额外4.5M；3个确认种子3M，共25.5M DiT更新，加共享VGAE/cache准备、Validation、报告和测速。单个完整训练每种权重有20个50k候选，screen每种5个；搜索成本和选优机会都必须与冻结baseline配方分开披露。先用目标设备preflight和第一轮实际日志估算GPU-hours，不将其他显卡的速度直接外推。

每个持久checkpoint包含raw、两套EMA与Adam状态，约为模型参数FP32字节数的5倍，另加buffer/RNG。默认容量的源模型约9.8M参数，单份checkpoint约200MB量级；较大模型按实测参数量增长。每种权重、每个monitor时点保存72份完整NPZ，磁盘也要为预测、参考与pre-boundary数组预留空间。不得覆盖早期checkpoint来节省空间；确需调整保留策略时先形成新的明确交付配置。

精确恢复的CPU验收比较了raw、EMA、Adam、训练generator和sample cursor。CUDA散射和attention kernels可能不满足位级确定性；集群运行记录源码、设备、环境和随机状态，按数值与科学指标验证复现。
