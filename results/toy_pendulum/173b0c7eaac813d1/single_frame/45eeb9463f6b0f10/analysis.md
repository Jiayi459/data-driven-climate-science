**Material Passport:** academic-research-suite / experiment-agent, validate mode; 2026-11-04; version single_frame_analysis_v1; Verification Status **ANALYZED**. 已核对返回的文件和对应代码，未重跑训练或从模型重新计算特征。整体解释置信度 CAUTION：只有一个训练 seed，缺少原始特征和模型文件。

**输入与核对范围。** 来源为 `/Users/haojiayi/Downloads/pendulum_results_173b0c7eaac813d1 (1)`，共34个结果文件（27 JSON、6 PNG、1 CSV），另有4个不参与分析的 `.DS_Store`。全部 JSON/CSV 已读取；6张图已查看，包括用户提供的 latent 图。数据集目录的4个结果文件与上次五帧实验完全相同。`l1_data.py`、`l1_learning.py`、`l1_evaluation.py` 的本地 SHA 与报告一致；scorecard 的 SHA 与 complete.json 一致，146条指标无重复。详见 [audit.json](audit.json) 和 [scorecard.csv](scorecard.csv)。

四个训练 run 均为 L1a、seed42、window=1、lag=5、hidden_dim=64、projection_dim=2、view_noise_std=0.01、run_tag=single_frame。主训练30epochs，NSV refinement60epochs。只有L1a的学习结果，不含L1b/L1c的新模型。测试集包含20条独立轨迹、1520个样本；latent图按测试索引每3个取1个，显示507个点。

**噪声增强。** 图片先从uint8除以255，得到约[0,1]的灰度值。训练及训练目标验证时，每个像素加独立的零均值高斯噪声，标准差0.01：`I_aug = I + 0.01 * Normal(0,1)`。两路增强独立，每个训练epoch改变噪声。不会改变模拟的theta/velocity，也不是随机力或随机改变摆角；不截断噪声后的像素值。目的是鼓励表示对轻微像素扰动稳健，效果不是由增强定义保证的。

InfoNCE/BT-temporal增强I_t和I_(t+5)，BT-same增强同一帧的两个视图，NSV只增强输入，未来目标不加这层训练增强。L1a的数据条件本身noise_std=0；L1b的观测噪声是另一项设置。保存评估特征时不使用这层增强，因此本次latent图的散点并非画图时加入0.01噪声造成。

**坐标和颜色。** 每列上下两排使用同一组点、同一组二维坐标，只把着色变量从theta换成velocity。坐标是训练集拟合的PCA变换后的表示数值，没有直接的角度、时间或速度单位；不同模型各自拟合PCA，其方向、符号、尺度不可直接跨模型对齐。颜色使用模拟真值，没有z-score或min-max归一化：angle为theta，单位rad；angular velocity为dtheta/dt，单位rad/模拟时间单位。本实验采用g=L_ref=1的时间尺度，不应直接解释成rad/s或rad/day。theta是摆杆相对竖直向下的角度，不是振荡相位；同一theta可以对应向左或向右运动。

**为什么二维z还做PCA。** InfoNCE/Barlow的路径是image→CNN encoder→h64→MLP projector→z2；NSV的z_d2来自冻结h后的refiner encoder，refiner decoder将其重建为h。z不是图像decoder的输出。绘图统一调用`PCA(min(2, dimension)).fit(train_features)`，然后transform测试样本。无whitening、无逐坐标标准化。

对前四列二维表示，`pc = (z - mean_train) @ Q`，Q为正交矩阵。因此是中心化加旋转/可能的镜像、轴交换，不减少维数，不损失信息，不会把直线变成U形或散点云。直接画z1/z2也可以。图框自动独立设置x/y范围，所以屏幕上的长宽比例还会夸大微小方向。最后一列的PCA基线实际使用64维特征，绘图再投影到2维，此处存在信息损失。虽然标签保留delay_pca，当前window=1并没有拼接历史；流程是单帧→16×16块平均→256维像素→训练拟合PCA64→展示PCA2。

**各列的证据。** 以下均为全部1520个测试样本的当前角度读出R²，而非只从图形判断。读出是在冻结表示上另行拟合的评估模型。

| 表示 | 线性ridge角度R² | MLP角度R² | 图形解释 |
|---|---:|---:|---|
| BT-same z2 | 0.478910 | 0.847874 | 弯折、局部靠近的曲线，角度关系比直线复杂；有限MLP比线性读出好。MLP达到200次迭代上限，不能当作最优非线性性能。 |
| BT-temporal z2 | 0.999335 | 0.998291 | 沿PC1有清晰角度顺序。PC1为1量级，PC2约0.001量级；整体近直线，不能根据放大后的波纹判定复杂二维流形。 |
| InfoNCE z2 | 0.983096 | 0.996215 | U形曲线，角度沿曲线有序。二维坐标足以嵌入一维角度变量；U形不表示物理二维状态。 |
| NSV z_d2 | -0.000315 | -0.002237 | 看似散点云，所测试的读出未恢复角度。不能把它解释为成功的二维物理状态，也不能仅凭图断言任何非线性读出都失败。 |
| PCA64基线 | 0.999999 | 0.993541 | 图显示PCA64再降到2维的弯曲投影；表内读出使用完整64维，不能将此分数归于图中两个PC。 |

BT-temporal z的训练子样本participation ratio=1.00000092，局部ID(k=10,20,40)=0.931/0.966/0.986，支持接近一维的解释。BT-same z的participation ratio≈1.974并不表示物理自由度为2：弯曲的一维曲线可以在两个正交方向都有较大方差。

四个encoder h64的线性角度R²分别为BT-same0.999069、BT-temporal0.999888、InfoNCE0.999978、NSV0.999631；局部ID均约1。NSV h保留了角度，当前refinement的z_d2未形成便于读出的角度表示。d2最佳验证重建MSE约0.879；d8曾到0.471但最后退化至0.882。其Sine refiner的强振荡/折叠和有限采样是散点外观的可能解释，缺少features/checkpoints无法定位原因。不能将此推测升级为已验证的优化根因。

**为什么速度颜色混杂。** 固定摆长的单帧渲染只依赖theta，无法区分相同位置的不同速度。训练时用两帧构造loss，不等于推理时同时输入两帧。一个固定的encoder仍只能输出单帧信息。四种h/z当前速度的ridge R²均约0，PCA64也约0。这符合观测限制，不能据此单独判定SSL失败。原始连续渲染的输入族是一维；量化、折叠和有限样本会影响ID估计。NSV z_d2的有限尺度ID≈2不构成物理状态恢复证据。

**另外几张图。** state_readout_test是单条轨迹18的当前状态监督线性读出：BT-temporal/PCA角度接近真值，InfoNCE角度极值略受压缩，BT-same部分恢复，NSVz_d2接近零；所有方法当前速度恢复弱。它不是自主时间积分，也不是decoder重建图。

forecast_test显示直接多步ridge读出，误差随周期位置不单调：BT-temporal z在12frames（0.24T0）的未来速度R²=0.98836，但未来角度R²≈0.00895；50frames（T0）未来角度R²=0.95361而速度R²≈0.02791。小振幅近谐振子关系解释这种现象：四分之一周期后的速度主要由当前角度决定，而那时的角度主要依赖当前速度。这是由动力学作出的近似解释，不证明从单帧恢复了当前速度或完整状态。oracle persistence使用真实当前速度，输入信息不同；阴影区间是轨迹bootstrap且条件于单个训练模型，未做方法间配对显著性检验。

NSV validation_prediction中的未来图像预测较模糊。MSE在同一位置对应多个可能未来时倾向条件均值，符合这种模糊的可能来源；单张预测图不能区分不可观测性和优化不足的贡献。该预测图使用NSV h→图像decoder，并不是z_d2解码结果。

**局限与复现状态。** 下载包含报告与图片，不含原始模拟数组、features.npz、evaluation_truth.npz、模型/refiner checkpoint。因此不能重新绘制原始z、检查密集theta扫描、精确复算绘图PCA或独立复现训练。所有数值为返回报告的读取和一致性检查，不是训练复现。四个MLP拟合报告收敛警告：BT-same z、NSV z_d4/z_d8、PCA64。单seed结果不能作为普遍方法排名。五帧与单帧实验改变了可观测信息，不能把速度读出下降称为单纯模型退步。

**统计误读检查：11/11适用性已检查。** Simpson：没有每轨迹重算，限于汇总描述。Ecological：1520帧不视为1520个独立实验，独立单位为20轨迹。Berkson：限定L1a、当前初值范围和留有预测长度的样本。Collider：未进行基于恢复成功的条件筛选，无因果结论。Base rate：连续回归对照R²≈0的均值预测水平。Regression to mean：不承诺下一次训练改善。Survivorship：只知返回的四个完成run，未知是否存在其他尝试。Look-elsewhere：多方法、多维度、多指标是探索比较，未声称经过多重检验的显著性。Forking paths：单帧改动及其历史背景明确，不作为预注册确认。Correlation/causation：可读出与恢复动力学机制区分。Reverse causality：当前监督readout不冒充未来预测；真实未来预测也不证明因果识别。

miao
