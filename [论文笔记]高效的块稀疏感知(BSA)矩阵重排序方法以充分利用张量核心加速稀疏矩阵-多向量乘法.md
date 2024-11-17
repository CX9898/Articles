![Accelerated Block-Sparsity-Aware Matrix Reordering for Leveraging Tensor Cores in Sparse Matrix-Multivector Multiplication](img/[论文笔记]高效的块稀疏感知(BSA)矩阵重排序方法以充分利用张量核心加速稀疏矩阵-多向量乘法/封面.png)

# [论文笔记]高效的块稀疏感知(BSA)矩阵重排序方法以充分利用张量核心加速稀疏矩阵-多向量乘法

**Accelerated Block-Sparsity-Aware Matrix Reordering for Leveraging Tensor Cores in Sparse Matrix-Multivector
Multiplication**

论文于2024年发表于Euro-Par 2024会议(30th International European Conference on Parallel and Distributed Computing).

稀疏矩阵-多向量乘法(SpMM)又称为稀疏矩阵-稠密矩阵乘法是深度学习模型和科学计算应用中的关键内核.
然后, 由于非零元素和不规则分布的内存访问模式, 实现高性能的SpMM具有挑战性.
在论文中, 提出了一种新颖的稀疏矩阵重排序算法, 该算法考虑了块稀疏性, 以增强Tensor Cores上SpMM的数据局部性.
对大量稀疏矩阵的实验结果表明了重排序算法的有效性以及利用Tensor Cores进行SpMM的好处.

---

## 引言

几个最先进的深度学习模型, 如卷积神经网络, 图神经网络和TransFormer, 在训练和推理阶段都执行大量的矩阵-矩阵乘法.
因此, 专门化的加速器, 如TPU(Tensor Processing Unit)和NVIDIA的Tensor Cores, 最近被引入来加速稠密矩阵-稠密矩阵乘法(DMM).

SpMM核心在GPU上存在基本的计算挑战. 首先, 左乘的稀疏矩阵和右乘的稠密矩阵中元素的不规则访问由左乘稀疏矩阵中非零元素的位置决定.
这种不规则的内存访问导致GPU上全局内存的带宽的低效使用和缓存命中率降低. 其次, 稀疏矩阵中非零元素的不规则分布导致负载不平衡问题和GPU上可利用的并行性的降低.

之前一些优化SpMM的努力主要集中在重新排列稀疏矩阵以提高数据局部性. 稀疏矩阵重新排序的主要目标是重新组织原始矩阵,
从重新排序矩阵中获得的密集块可以用来与右乘矩阵进行稠密矩阵-稠密矩阵乘法.

论文的主要目标是开发一种新颖的加速块稀疏感知(Block-Sparsity-Aware)重排算法. 用于在GPU上高效的重排不规则稀疏矩阵中的行,
旨在提取高密度块并明智地利用Tensor Cores进行SpMM.

为了克服基于列索引的聚类问题, 论文的矩阵重排算法首先将行划分为多个列块, 以识别块级稀疏模式, 以识别块稀疏模式.
为了增强SpMM的数据局部性, 在行聚类过程中, BSA重排算法不仅考虑了非零列块的位置, 还考虑每个列块中非零项的数量.
为了有效地测量编码行向量之间的相似度, 采用考虑向量实际的加权Jaccard相似度.

> 加权Jaccard相似度:

在对行进行重排序后, 根据密度阈值将重排序后的矩阵拆分为稠密块和稀疏剩余.
采用Blocked-Ellpack格式来储存稠密块中的元素, 并利用NVIDIA的cuSPARSE Block-SpMM来加速运算稠密矩阵.
对于稀疏剩余部分, 使用压缩稀疏行(CSR)格式来储存, 并在常规CUDA核心上使用NVIDIA的cuPARSE进行SpMM操作.

> Blocked-Ellpack格式: Blocked-Ellpack格式是ELLPACK格式的一个变种, 将矩阵划分为多个块, 每个块内部使用ELLPACK格式储存.
> ELLPACK格式是一种储存稀疏矩阵的压缩格式, 它将每一行的非零元素按列索引排序后储存.

使用来自深度学习矩阵集合的2586个稀疏矩阵进行广泛的比较和评估,
结果表明,论文的并行SpMM实现(称为BSA-SpMM)比最先进的替代方案实现了高达21.99倍的加速.

---

## 背景及相关工作

### 稀疏矩阵-多向量乘法(SpMM)和稀疏矩阵表示

### 带有Tensor Cores的核心的图形处理器(GPU)

### 关于稀疏矩阵重排以优化稀疏矩阵乘法(SpMM)的相关工作

---

## BSA-SpMM: 块稀疏感知矩阵乘法

### BSA-SpMM概述

### BSA-SpMM的细节

---

## 实验评估

### 实验设置

### 性能评估

---

## 结论

---