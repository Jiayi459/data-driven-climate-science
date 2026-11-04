# L1a 实际运行的架构、样本量和超参数

本文件记录 2026-10-28 已完成的五帧历史实验。2026-11-04 用户已将后续默认输入改为单帧；本文的五帧配置和数值保留为历史记录，不代表当前 notebook 默认值。

依据：dataset `173b0c7eaac813d1`、四份 `run.json`、`evaluation.json` 和哈希匹配的本地实现。以下描述这次已经运行的配置，不是建议的下一版配置。网络仅为核对层尺寸/参数量而实例化，没有重新训练。

**先区分三个尺寸**

- `64×64`：每一帧灰度图的空间尺寸。
- `[5,64,64]`：一个输入样本，五帧按 channel 堆叠。
- `h=[64]`：encoder 输出，64 个数，不是 64×64 网格。
- `z=[2]`：本轮 InfoNCE/BT 的 projector 输出；NSV refiner 则分别输出 d=1/2/3/4/8。

神经网络权重经各自训练得到，各方法不共享训练后的参数；InfoNCE/BT 同一个模型内部的两个分支共享参数。Delay-PCA 通过数据拟合线性变换。绘图另用训练集拟合 PCA 展示，所以图形还受到展示坐标的影响。

**模拟数据及样本量**

| 配置 | 本次值 |
|---|---|
| 条件 | L1a，固定摆长、无观测噪声 |
| g / L / damping | 1 / 1 / 0.02 |
| 时间长度 | 每条 8T0，T0=2π√(L/g)=2π |
| 采样 | 每周期 50 个间隔，dt=T0/50≈0.12566；包括初始时刻共 401 帧 |
| 数值积分 | RK4，每个观测间隔 8 个子步 |
| 初始条件 | θ∈[-1,1]、v∈[-0.5,0.5] 均匀抽样；初始状态范数<0.15 时重抽 |
| 数据 seed / 划分 seed | 20261027 / 20261044 |
| 图像 | 64×64 灰度，uint8 缓存，读取时转 float32 并除以 255 |
| 轨迹数 | 100，整条轨迹划分为 train60 / val20 / test20 |
| 原始帧数 | 总共 40,100；train24,060 / val8,020 / test8,020 |
| 训练增强 | Gaussian std=0.01；无裁剪、翻转、逐帧标准化或时间重排 |

| 使用阶段 | 每条轨迹窗口数 | Train | Val | Test |
|---|---:|---:|---:|---:|
| 第一阶段 representation 训练的候选配对，stride=2 | 196 | 11,760 | 3,920 | 不用于训练或选 checkpoint |
| 统一特征导出、NSV refinement、probe，stride=4 | 75 | 4,500 | 1,500 | 1,500 |

第一阶段窗口末帧为 4,6,…,394，预留 lag=5；BT-same 虽不需要未来帧，也沿用同一候选索引。统一特征导出末帧为 4,8,…,300，为最大 100 帧预测 horizon 预留未来数据。因此第二行样本更少，不是数据丢失。

第一阶段 batch128 且 `drop_last=True`：每个 epoch 实际使用 91 个 train batch=11,648 对，30 个 val batch=3,840 对。训练打乱后丢弃 112 对，具体哪些随 epoch 改变；验证按固定顺序丢弃最后 80 对。每个模型共 2,730 次第一阶段 optimizer updates。Refiner 不丢弃尾 batch，每 epoch 36 个 train batch，共 2,160 次 updates/维数。

窗口有时间重叠，11,760 个窗口不等于 11,760 条独立轨迹。独立训练轨迹只有 60 条。39 另生成了 300 条变摆长轨迹，但本次四个 L1a run 没有使用它们。

**共同 CNN encoder**

下列 shape 省略 batch 维度；默认输入 batch shape 为 `[128,5,64,64]`。

| 层 | 设置 | 输出 shape |
|---|---|---|
| 输入 | 5 帧灰度历史，作为 5 个通道 | 5×64×64 |
| Conv1 | 5→16，kernel5，stride2，padding2；GroupNorm(4 groups)+SiLU | 16×32×32 |
| Conv2 | 16→32，kernel3，stride2，padding1；GroupNorm(4 groups)+SiLU | 32×16×16 |
| Conv3 | 32→64，kernel3，stride2，padding1；GroupNorm(4 groups)+SiLU | 64×8×8 |
| Flatten | 64×8×8=4096 | 4096 |
| Linear | 4096→64 | h：64 |

每个 encoder 有 **287,584** 个可训练参数。四种方法用相同结构、相同 seed 的初始化，随后分别训练。输入 θ/v/L 数值不进入 CNN；它们用于模拟和事后 probe。

**第一阶段共同训练设置**

| 参数 | NSV / InfoNCE / BT-temporal / BT-same |
|---|---|
| optimizer | AdamW |
| learning rate | 0.001，恒定 |
| weight decay | 0.00001 |
| batch size / epochs | 128 / 30 |
| seed | 42，各方法只跑一个 seed |
| gradient clipping | global norm 上限 5.0 |
| train / val loader | train shuffle=True；val shuffle=False；两者 drop_last=True；num_workers=0 |
| 输入历史 / 训练 stride / 导出 stride | 5 / 2 / 4 帧 |
| checkpoint | 每轮计算验证 loss，保留最低值；30 轮结束后恢复最佳权重 |
| 训练提前停止 / 学习率调度 | 本次没有；全部运行完 30 epochs |
| view noise | std=0.01；训练每个 epoch 改变增强，验证使用固定增强 |

AdamW 其余参数使用运行环境默认值，未在 run.json 中逐项显式配置。运行元数据记录 Python3.13.16、PyTorch2.11.0+cu130、NumPy2.1.3、device=cuda；没有记录实际 GPU 型号。

**1. NSV：预测编码 + 独立低维压缩**

第一阶段：

```text
输入 [I(t−4),…,I(t)]，5×64×64
  → 共同 CNN → h(64)
  → Linear64→4096 → reshape64×8×8
  → bilinear 上采样×2 → Conv64→32(k3,p1) → SiLU：32×16×16
  → bilinear 上采样×2 → Conv32→16(k3,p1) → SiLU：16×32×32
  → bilinear 上采样×2 → Conv16→5(k3,p1)：5×64×64
  → 预测未来 [I(t+1),…,I(t+5)]
```

Upsample 的 `align_corners=False`，decoder convolutions 的 stride=1。输出层没有 sigmoid，以允许观测值超出[0,1]。

- 第一阶段训练/验证候选配对数：11,760 / 3,920。
- loss：未来五帧的像素 MSE；g/L/θ/v 不进入 loss。
- 输入加 std0.01 增强；预测 target 不额外加这种增强。L1a target 因观测噪声=0 而为干净帧，L1b 会预测带观测噪声的 target。
- 使用共同 AdamW 设置；最佳第一阶段 epoch=29。
- 图像 decoder 参数=290,053；第一阶段总参数=577,637。

第二阶段冻结第一阶段模型，使用导出的 h：

```text
h(64) → 使用训练集 mean/std 标准化
      → Sine64→128 → Sine128→64 → Linear64→d → z_d
      → Sine d→64 → Sine64→128 → Linear128→64
      → 重建标准化的 h
```

| refinement 参数 | 本次值 |
|---|---|
| d | 1、2、3、4、8，五个独立 refiner |
| 训练 / 验证 h 数 | 4,500 / 1,500 |
| optimizer / lr | Adam / 0.001 |
| weight decay | 0；这里没有使用第一阶段的 1e-5 |
| batch / epochs | 128 / 每个 d 60 轮；尾 batch 保留 |
| seed | 42+d，即43、44、45、46、50 |
| loss | 标准化 h 的重建 MSE |
| Sine 激活 | sin(30·Linear(x))，ω0=30 |
| 梯度裁剪 / 学习率调度 | 本次均未设置 |
| checkpoint | 使用最低验证 MSE 的权重 |
| 每个 d 的参数数 | 33,345 / 33,474 / 33,603 / 33,732 / 34,248 |

标准化 std 下限为 max(最大坐标 std 的1%,1e-5)。encoder 与 decoder 的首个 SineLayer 都设为 first=True，权重初始化范围±1/n_in；后续 SineLayer 范围±√(6/n_in)/30；两处 Linear 输出采用默认初始化。这里是本地适配版，应与 Chen 官方结构区分。

保存的 representation 是 `h`、`z_d1`、`z_d2`、`z_d3`、`z_d4`、`z_d8`。本次图中只选中了 `z_d2`；最佳 refiner epochs 分别为59、30、40、10、36。

**2. InfoNCE**

```text
A、B 两个窗口，各自经过同一个网络：
5×64×64 → 共同 CNN → h64
        → Linear64→128 → LayerNorm128 → SiLU → Linear128→2 → z2
        → L2 normalize → z_unit2，用于 cosine contrastive loss
```

- 训练/验证候选配对数：11,760 / 3,920；使用共同 AdamW、batch128、30epochs 等设置。
- positive：同轨迹 A=[t−4,…,t]，B=[t+1,…,t+5]；两侧独立添加 std0.01 noise。
- negative：batch 内其他 cross-view 配对；同轨迹 anchor 时间差<10帧的非对角配对被排除。
- loss：两个方向的 cross-entropy 平均；temperature=0.1。
- 总参数=296,418，其中 projector=8,834；两个分支共享参数，不将参数量乘以2。
- 最佳验证 epoch=12；保存 `h(64)`、原始 `z(2)`、归一化 `z_unit(2)`。
- 图中是原始 z 的二维展示；loss 中的 z_unit 位于圆周，只有一个连续自由度。

**3. Barlow Twins temporal**

```text
两侧共享：5×64×64 → 共同 CNN → h64
          → Linear64→128 → LayerNorm128 → SiLU → Linear128→2 → z2
```

- 训练/验证候选配对数：11,760 / 3,920；共同 AdamW、batch128、30epochs。
- positive views：与 InfoNCE 相同的相邻五帧窗口，lag=5；两侧独立 noise std0.01。
- 不构造 negatives。
- 每侧 z 按 batch 的各坐标减均值、除以 sqrt(var+1e-5)，var 使用 unbiased=False。
- C=两侧标准化输出的 cross-correlation；loss=Σ_i(C_ii−1)²+0.005Σ_(i≠j)C_ij²。
- 冗余项 λ=0.005，稳定项 ε=1e-5；不使用 InfoNCE temperature 或逐样本 L2 normalize。
- 总参数=296,418；最佳验证 epoch=28；保存 `h(64)`、`z(2)`。

**4. Barlow Twins same-time**

```text
两侧共享：5×64×64 → 共同 CNN → h64
          → Linear64→128 → LayerNorm128 → SiLU → Linear128→2 → z2
```

- 训练/验证候选配对数：11,760 / 3,920；共同 AdamW、batch128、30epochs。
- 两侧来自同一段 [t−4,…,t]，只是独立 noise std0.01；实际配对时间差=0。
- run.json 的 lag=5 仍用于统一候选窗口范围，**并不意味着 BT-same 的第二个 view 移动了5帧**。
- 与 BT-temporal 相同的 cross-correlation loss：λ=0.005、ε=1e-5，无 negatives。
- 总参数=296,418；最佳验证 epoch=6；保存 `h(64)`、`z(2)`。

InfoNCE、BT-temporal、BT-same 的架构完全相同；其 pattern 差异来自配对、目标函数及训练后权重，而不是给它们指定了不同的 CNN 层数。

**5. Delay-PCA**

```text
5×64×64 图像历史
  → 每帧按4×4像素块求均值 → 5×16×16
  → Flatten → 1280维
  → 训练集拟合 PCA → 64维
  → 绘图另投影到2维
```

- 无神经 encoder/projector/decoder，也没有 optimizer、learning rate、epochs 或 pair loss。
- 从统一索引的4,500个训练窗口中随机选择3,000个拟合 PCA。
- `n_components=64`、`svd_solver='randomized'`、`random_state=123`；使用默认不 whitening 的 PCA。
- 使用 L1a 原始观测：噪声0，不加神经网络训练时的 std0.01 view augmentation。
- 将全部 train/val/test 窗口变换后用于相同 probe；64成分累计解释方差约99.833%。
- PCA 的主成分由数据拟合，其二维展示不等于只使用两个成分预测。

**事后评估的超参数：与 representation 训练分开**

| 项目 | 本次配置 |
|---|---|
| probe 训练/验证/测试 | 4,500 / 1,500 / 1,500 个窗口 |
| Ridge | 特征、目标尺度只用 train 拟合；α∈{0.0001,0.01,1,100,10000}，以 validation normalized MSE 选择 |
| 非线性 probe | 输入表示→Dense64(tanh)→θ/v；MLPRegressor，L2 α=0.01，lr=0.001，batch256，max_iter200，seed123，early_stopping=False |
| 非线性 probe 使用范围 | 当前状态；未来预测只用 ridge |
| 预测 horizon | 0、12、50、100 帧，即0、0.24T0、1T0、2T0 |
| 误差区间 | 按轨迹 bootstrap300次，seed123 |
| ID | training-only；k=10/20/40，最多2000点，排除同轨迹邻居；nb42 seed123，run内部 seed42，oracle默认seed17 |
| 绘图 | PCA仅在训练表示上拟合；展示每隔3个test窗口，共500点 |

图中 `h=64`、projector 内部宽度128、probe隐藏层64、图像64×64分别是不同对象，不能混为同一个 hidden state。

miao
