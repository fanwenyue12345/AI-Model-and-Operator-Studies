# 01 — ReWY 项目研究

> 项目：ReWY  
> 官方仓库：https://github.com/flashserve/ReWY  
> 论文：https://arxiv.org/abs/2611.01940  
> 当前记录状态：从原 AI-Infra-Knowledge-Base 迁移的初步研究笔记；数学推导与硬件实验尚未独立完整复现。

## 1. 项目定位

ReWY 研究 Delta-Rule Attention 家族的反向传播计算优化，涉及 DeltaNet、Gated DeltaNet（GDN）、Kimi Delta Attention（KDA）和 GDN2。

我们重点关注：
- Residual WY：利用局部 WY 结构与伴随变量重组反向计算；
- Matrix Intra：重新组织 KDA/GDN2 中的向量门控交互；
- Gram Elimination：避免 GDN 反向过程中的冗余 Key Gram Matrix 计算；
- FLA 接入、GPU/NPU Kernel 实现及性能验证。

## 2. 已迁移的研究记录

- [ReWY 初步研究记录](Research_Notes.md)：项目背景、伴随变量推导框架、三项优化、个人理解、待研究问题和初步验证计划。

该文档从原仓库的 `02_Our_Research_Notes.md` 迁移，保留原有内容与研究状态。后续再按照本仓库的统一学习框架拆分和完善，不在迁移时擅自改变原有研究结论。

## 3. 后续学习目录规划

| 层级 | 后续需要补充的内容 |
|---|---|
| 01 Background | Delta-Rule Attention 的发展背景与 ReWY 要解决的问题 |
| 02 Paper & Algorithm | 完整论文推导、Residual WY、Matrix Intra、Gram Elimination |
| 03 Architecture | ReWY 的软件架构、模块关系和数据流 |
| 04 Source Code | 从 FLA API 到 Autograd、算子和 Kernel 的调用链 |
| 05 Operators & Kernels | 前向/反向算子、矩阵乘法、Kernel 实现 |
| 06 Hardware & Optimization | GPU/NPU 映射、内存、并行与编译 |
| 07 Benchmark & Experiments | 正确性、梯度误差、性能与消融实验 |
| 08 Research Ideas | 可推广的伴随结构、自动优化和跨硬件抽象 |

以上是后续学习计划，不代表这些工作已经完成。

## 4. 与通用优化理论的关联

ReWY 是多个优化方向的具体案例：

- 算法与数学优化：伴随变量、线性方程、梯度重组；
- 计算图优化：共享中间量、减少冗余梯度计算；
- Kernel 优化：将重组后的计算映射为 GPU/NPU Kernel；
- 数值精度优化：验证浮点计算下的梯度和输出误差。

通用理论请参见 [AI-Infra-Knowledge-Base](https://github.com/fanwenyue12345/AI-Infra-Knowledge-Base)。

## 5. 原始资料

- [ReWY GitHub](https://github.com/flashserve/ReWY)
- [ReWY 论文](https://arxiv.org/abs/2611.01940)
- [ReWY 架构说明](https://github.com/flashserve/ReWY/blob/main/docs/architecture.md)
- [ReWY 实验结果说明](https://github.com/flashserve/ReWY/blob/main/docs/results.md)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
