# AI Model and Operator Studies

个人 AI 模型、Attention 算法与高性能算子项目学习知识库。

本仓库按**具体项目**组织学习记录，目标是从研究背景出发，逐步理解论文、数学原理、模型架构、源码、算子实现、硬件执行和实验验证。

## 项目目录

| 项目 | 研究主题 | 状态 |
|---|---|---|
| [01 — ReWY](01_ReWY/README.md) | Delta-Rule Attention 反向传播优化、伴随方法与 Kernel 实现 | 已迁移初始研究笔记，持续完善 |

后续计划研究：FLA、HiLS-Attention、FlashAttention、NSA、vLLM、SGLang 等。

## 每个项目的统一学习框架

项目内容逐步按以下层次建立，不要求一开始就填满所有文档：

1. **Background & Motivation**：历史背景、研究问题、相关工作。
2. **Paper & Algorithm**：论文阅读、数学原理、核心公式与复杂度。
3. **Architecture**：整体架构、模块关系、数据流。
4. **Source Code**：源码目录、关键类、函数调用和张量形状。
5. **Operators & Kernels**：算子拆分、Kernel 实现、数据布局和并行执行。
6. **Hardware & Optimization**：GPU 架构、内存、编译器、性能优化与硬件适配。
7. **Benchmark & Experiments**：正确性、数值误差、性能基线和实验结果。
8. **Research Ideas**：未解决问题、研究假设、实验设计和潜在创新。

## 记录原则

- 区分论文/官方源码可以确认的事实、个人理解和待验证假设。
- 公式说明符号、推导、张量维度及适用条件。
- 不把项目作者的成果写成个人创新。
- 不把计划中的实验写成已经完成的结果。
- 性能结论必须注明硬件、软件环境、输入规模、精度和计时边界。
- 尽可能保留原始论文、代码和官方文档链接。

## 与通用优化知识库的关系

本仓库记录具体项目；通用优化理论放在另一个仓库：

[AI-Infra-Knowledge-Base](https://github.com/fanwenyue12345/AI-Infra-Knowledge-Base)

例如，ReWY 的具体推导和项目实现放在本仓库；伴随方法、计算图优化、Kernel 优化等通用理论仍归入 AI-Infra-Knowledge-Base。

**研究主线：** Background → Mathematics → Architecture → Source Code → Operator/Kernel → Hardware → Benchmark → Research
