# 我们的研究记录：从 ReWY 理解 Attention 优化

> 本文记录我们对 ReWY 的学习过程、个人理解和待验证问题。它不是 ReWY 原论文的替代，也不把原作者已经完成的工作归为我们的创新。
>
> **记录状态：** 初步学习与概念分析；尚未独立复现数学推导和硬件实验。

## 1. 项目基本信息

- 项目：ReWY
- 仓库：https://github.com/flashserve/ReWY
- 论文：https://arxiv.org/abs/2611.01940
- 论文标题：*ReWY: Faster Backpropagation for Delta-Rule Attention — From 7 to 3 Matrix Multiplications*
- 研究对象：DeltaNet、Gated DeltaNet（GDN）、Kimi Delta Attention（KDA）和 GDN2 的反向传播。
- 核心目标：在不改变模型递推和训练目标的前提下，减少 Delta-Rule Attention 家族反向传播中的冗余计算。
- 项目实现：算子实现、FLA 接入、测试、实验复现工具；包含 NVIDIA GPU 路径和实验性的 Huawei Ascend NPU 路径。

## 2. ReWY 在八类优化体系中的位置

ReWY 不是只能归入一种优化类别。按主要贡献和工程实现区分：

| 层次 | 归属 | 在 ReWY 中的体现 |
|---|---|---|
| 主要：01 算法与数学优化 | Algorithmic & Mathematical Optimization | 利用线性方程和伴随方法重组梯度计算 |
| 次要：02 计算图优化 | Computation Graph Optimization | 消除部分中间梯度路径，复用共享伴随量，减少局部矩阵乘法 |
| 工程实现：05 Kernel 优化 | Kernel Optimization | 将优化后的计算组织为 GPU/NPU Kernel |
| 相关：07 数值精度优化 | Numerical Precision Optimization | 保持既有 FP32 中间量的精度约束，并验证浮点实现 |

分类原则：**数学推导是主要创新来源，计算图重组是直接优化机制，Kernel 与硬件适配负责把数学收益转化为实际性能。**

## 3. 研究背景：为什么要优化反向传播？

训练神经网络通常包括前向传播、损失计算、反向传播和参数更新。反向传播需要计算损失对输入、中间状态和模型参数的梯度。

Delta-Rule Attention 使用递归状态更新，并通过 Chunkwise 计算将块内依赖组织为矩阵计算。其反向传播中存在多个中间矩阵与伴随梯度。

研究问题是：原有反向计算图中是否存在可以合并、复用或消除的中间计算？

ReWY 的路线不是改变模型的递推规则，而是重新推导特定局部计算结构的梯度。

## 4. 我们重点讨论的数学思路：伴随变量如何出现？

### 4.1 从局部线性方程出发

在我们讨论的局部 WY 计算关系中，使用如下形式表达：

$$
MR = Z - ES
$$

其中：

- $M$：具有单位下三角结构的矩阵；
- $R$：局部修正变量；
- $Z$：输入相关的矩阵项；
- $E$：门控/衰减相关矩阵项；
- $S$：块入口状态。

令损失函数依赖于 $R$：

$$
\mathcal L = \mathcal L(R)
$$

定义：

$$
\bar R = \frac{\partial \mathcal L}{\partial R}
$$

这里的横线表示反向传播中的梯度（伴随量），不是普通代数变量。

### 4.2 对定义 R 的方程做微分

对两侧求微分：

$$
(dM)R + M(dR) = dZ - (dE)S - E(dS)
$$

整理：

$$
M(dR) = dZ - (dE)S - E(dS) - (dM)R
$$

令 $A=M^{-1}$，则：

$$
dR=A[dZ-(dE)S-E(dS)-(dM)R]
$$

### 4.3 用矩阵内积把共同项提取出来

使用 Frobenius 内积：

$$
\langle X,Y\rangle_F=\operatorname{tr}(X^TY)
$$

损失微分为：

$$
d\mathcal L=\langle \bar R,dR\rangle_F
$$

代入 $dR$，并使用：

$$
\langle X,AY\rangle_F=\langle A^TX,Y\rangle_F
$$

得到：

$$
d\mathcal L =
\left\langle A^T\bar R,
dZ-(dE)S-E(dS)-(dM)R
\right\rangle_F
$$

此时定义共享伴随变量：

$$
\boxed{P=A^T\bar R}
$$

因为 $A=M^{-1}$，也可以写成：

$$
\boxed{M^TP=\bar R}
$$

这个形式更有工程意义：实现时可以通过求解转置线性系统得到 $P$，不一定需要显式形成 $M^{-1}$。

### 4.4 为什么 P 能统一多个梯度？

将损失微分按变量分组，可得到：

$$
\bar Z=P
$$

$$
\bar E=-PS^T
$$

$$
\bar S=-E^TP
$$

对于 M：

$$
\bar M=-PR^T
$$

但 M 具有单位下三角结构，实际只允许其自由参数对应的梯度参与更新。因此需要投影到相应结构；在这里可写为：

$$
\bar M=-\mathcal L_{<}(PR^T)
$$

其中 $\mathcal L_{<}$ 表示提取严格下三角部分。

**直观理解：** 原来不同梯度路径中反复出现的线性变换，可以通过同一个伴随变量 $P$ 共享。随后利用 $P$ 计算多个输入项的梯度。

### 4.5 这不是“猜出一个矩阵”

从上述推导可以看出，$P$ 是由隐式微分和矩阵内积的伴随关系自然得到的。它不是任意试出来的经验参数。

研究者可遵循的思考过程是：

1. 找到定义中间变量的线性方程；
2. 对方程直接微分；
3. 将微分代入损失函数；
4. 利用伴随算子把线性变换从微分项转移到梯度项；
5. 定义共享伴随变量；
6. 按照变量分组整理梯度；
7. 再利用三角结构、对称性和矩阵乘法复用优化实现。

**边界说明：** 上面是我们为理解伴随变量而整理的局部推导框架。ReWY 的完整论文计算图还包含特定的 WY 结构、门控项和梯度路径。不能仅凭这个简化推导就声称已经独立复现论文中“7 次变 3 次”的全部计数；完整计数应回到论文公式、算法和代码逐项核对。

## 5. ReWY 的三项主要优化

| 方法 | 目标 | 初步理解 |
|---|---|---|
| Residual WY | 减少局部 WY 反向矩阵乘法 | 通过伴随量和残差关系重组梯度计算，局部矩阵乘法从 7 次降为 3 次 |
| Matrix Intra | KDA/GDN2 的向量门控交互 | 将相关向量交互重新组织为矩阵收缩 |
| Gram Elimination | GDN 中冗余 Key Gram Matrix | 避免显式构造冗余 Gram 矩阵，利用已有梯度关系 |

这些改进解决不同位置的计算问题。组合后的收益不能简单地把各个组件的加速比相乘或相加。

## 6. 我们对“优化思路”的初步认识

ReWY 提醒我们，优化一个算子不应该一开始就进入 CUDA/Triton 调参。

一个更系统的研究顺序是：

1. **理解数学结构：** 原计算究竟表达什么？
2. **检查依赖关系：** 哪些变量是中间量，哪些是真正需要的输出？
3. **重新推导：** 是否能通过代数、伴随或隐式微分改变计算组织？
4. **估算代价：** 矩阵乘法次数、FLOPs、内存读写和临时空间如何变化？
5. **映射到硬件：** 新计算图是否适合 Tile、Warp、Tensor Core 和流水调度？
6. **验证：** 梯度正确性、数值误差、显存和真实运行时间是否改善？

数学上等价不意味着浮点结果完全相同，也不意味着硬件执行必然更快。必须分别验证数学正确性、数值正确性和性能收益。

## 7. 值得继续研究的问题

以下是我们的研究问题，不是 ReWY 已经解决的结论。

### 问题 A：伴随结构能否推广到其他递归 Attention？

统一考虑状态更新：

$$
S_{t+1}=F(S_t,X_t;\theta)
$$

以及输出映射：

$$
Y_t=G(S_t,X_t;\theta)
$$

研究不同递归注意力算法的反向状态更新，寻找可共享的伴随计算结构。

候选对象：DeltaNet、GDN、KDA、GDN2，以及其他具有递归状态的序列模型。

### 问题 B：能否自动发现冗余梯度计算？

研究符号代数、计算图分析、编译器 IR 和自动微分之间的联系。目标是识别重复的伴随表达、公共矩阵乘法和可消除的中间变量。

### 问题 C：数学最优与硬件最优是否一致？

减少矩阵乘法不一定意味着端到端 Kernel 更快。新的计算图可能增加寄存器压力、降低 Occupancy 或改变内存访问模式。

需要联合考虑：

- FLOPs 与矩阵乘法规模；
- HBM IO 与片上数据复用；
- Register / Shared Memory 使用量；
- Warp、CTA 数量与 Occupancy；
- Kernel Launch 和同步开销。

### 问题 D：如何设计跨硬件的统一反向算子抽象？

希望数学接口保持一致，同时允许 CUDA、Triton 和 Ascend 后端采用各自最优的调度与数据布局。

## 8. 初步验证计划

| 阶段 | 工作 | 验证结果 |
|---|---|---|
| 1. 数学复核 | 对照论文完整推导 Residual WY | 公式、维度和矩阵结构一致 |
| 2. 源码追踪 | 从 FLA API 跟踪到 Autograd 和 Kernel | 得到真实调用链 |
| 3. 正确性 | 比较原始与优化实现的输出及梯度 | 误差满足设定容差 |
| 4. 性能 | 对不同序列长度、Batch、Head 和维度测试 | 延迟、吞吐和显存数据 |
| 5. 分析 | 分析瓶颈及硬件差异 | 解释性能变化的原因 |
| 6. 扩展 | 尝试其他递归 Attention 或硬件 | 形成可验证的新方法 |

## 9. 原始资料

- [ReWY GitHub 仓库](https://github.com/flashserve/ReWY)
- [ReWY 论文（arXiv:2611.01940）](https://arxiv.org/abs/2611.01940)
- [ReWY 架构说明](https://github.com/flashserve/ReWY/blob/main/docs/architecture.md)
- [ReWY 实验结果说明](https://github.com/flashserve/ReWY/blob/main/docs/results.md)
- [Flash Linear Attention（FLA）](https://github.com/fla-org/flash-linear-attention)

---

## 更新记录

- 2026-11-27：建立初版，记录 ReWY 的研究背景、伴随变量推导、三项优化、个人理解和待研究问题。
