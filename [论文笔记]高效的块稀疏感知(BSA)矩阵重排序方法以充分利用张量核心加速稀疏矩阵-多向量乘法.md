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

SpMM内核将M × K的稀疏输入矩阵S与大小为K × N的稠密输入矩阵D相乘, 生成大小为M×N的稠密输出矩阵O. 即O = S × D.

稀疏矩阵可以表示为基于块的稀疏格式, 例如Blocked-Ellpack格式和Variable Block Row(VBR)格式.

Blocked-Ellpack格式的主要优势在于它可以有效利用Tensor Cores.
特别是使用Blocked-Ellpack格式时, 可以通过使用NVIDIA的cuSPARSE库的 `cusparseSpMM()` 函数来利用Tensor Cores进行并行块矩阵乘法.

VBR格式与Blocked-Ellpack格式不同的是它会储存不同大小的非零块.
而NVIDIA的cuBLAS库的 `cublasGemmEx()` 函数支持可变的矩阵大小, 可以通过这个函数以利用Tensor Cores进行稀疏矩阵-稠密矩阵乘法.

### 带有Tensor Cores的核心的图形处理器(GPU)

Tensor Cores是NVIDIA开发的一种专门用于加速矩阵-矩阵乘法和累加的硬件单元.
在Volta架构中首次引入, 并在后续的Ampere架构和Hopper架构中得到了进一步的优化.
为了高效执行矩阵-矩阵乘法和累加操作, 通过32个线程的线程束(warp)协作.
与标准单精度(FP32)浮点格式相比, Tensor Cores通过利用低精度浮点格式(例如FP16), 实现了更高的性能和更低的内存需求.

### 关于稀疏矩阵重排以优化稀疏矩阵乘法(SpMM)的相关工作

为了增强SpMM的数据局部性, 最近已经开发除了集中稀疏重排序和压缩算法.
以往基于利用Tensor Cores, 为稀疏矩阵重排序的努力大致可以分为两类.

Hong等人提出了一种自适应稀疏分块(ASpT)方法, 首先将稀疏矩阵划分为多个行面板, 其中每个行面板由连续的行组成.
然后, 根据列密度对每个行面板的列进行重排, 形成密集分块.
Jiang等人进一步扩展了ASpT方法, 引入了一种行重排技术, 称为ASpT-RR.
ASpT-RR不是直接将稀疏矩阵的连续行分为行面板, 而是首先对行进行重排, 将相似的行分组到同一个行面板中.
但是当非零元素的列索引广泛分散时, 基于非零元素的列索引对行进行重排可能导致聚类失败.
Gale等人提出了一种行交换技术(称为Sputnik), 根据每行的非零元素数量重新排列行, 以实现GPU上常规SM中的负载均衡.

> 聚类(clustering)是指将具有相似特征的元素放在一起, 以提高缓存利用率和计算效率. 聚类失败意味着由于非零元素的列索引过于分散,
> 无法通过重排来形成连续的块, 从而无法利用处理器的向量化指令或其他优化手段来提高性能, 可能导致计算效率降低.

Labini等人提出了一种称为1-SA的行重排序技术, 该技术使用一种基于Saad算法的变体, 通过一维分块对行进行重排序.
1-SA将行划分为多个列分区, 并基于非零列分区模式的Jaccard相似度对行进行聚类.
使用VBR格式储存包含非零元素的变大小块, 使用NVIDIA的 `cublasGemmEx()` 函数利用Tensor Cores将这些块与右乘稠密矩阵相乘.
但是1-SA方法重新排序的矩阵中稀疏填充的块即使只包含一个非零元素, 也是以VBR格式储存, 然后传递给Tensor Cores进行计算.
Tensor Core上处理这些稀疏块会因为每个块中填充的零数量较多而导致利用率不足.
Yuke等人提出了一种稀疏图转换方案(称为TC-GNN), 将稀疏矩阵的行面板中的非零元素压缩, 以利用Tensor Cores进行SpMM操作.
但是在对行面板中非零元素进行压缩时, TC-GNN忽略了考虑各列中非零元素数量的变化, 并且没有比较行之间的稀疏模式.

---

## BSA-SpMM: 块稀疏感知矩阵乘法

在本节中, 首先描述BSA-SpMM的概述. 然后详细介绍块稀疏感知(BSA)重排序算法.

### BSA-SpMM概述

![BSA-SpMM概述图](img/[论文笔记]高效的块稀疏感知(BSA)矩阵重排序方法以充分利用张量核心加速稀疏矩阵-多向量乘法/BSA-SpMM概述图.png)
BSA-SpMM概述图

如图所示, 将输入稀疏矩阵S进行BSA重排, 生成的重排稀疏矩阵Sr进行分块(Tiling),
根据分块后的块(Tiled)的密度将块分为两类: **稠密块Sd**和**稀疏剩余Ss**.

将稠密块Sd以Blocked-ELL格式储存, 在Tensor Cores上与稠密矩阵D进行矩阵乘法运算.
将稀疏剩余Ss以CSR格式储存, 在常规CUDA核心上与稠密矩阵D进行矩阵乘法运算.

最终将两个结果合并后生成最终的输出矩阵O.

### BSA-SpMM的细节

#### BSA重排序中的相似性度量

#### 密集矩阵块的确定

#### 压缩Blocked-ELL格式以降低复杂性.

---

## 实验评估

- CPU: 12th Gen Intel(R) Core(TM) i7-12700 (12个物理核心)
- GPU: NVIDIA RTX 3080 GPU (69个安培SM, 计算能力为8.6, 10GB显存, 带宽为760GB/s).
- NVCC 12.1编译CUDA, 采用O3优化. C++11标准.

- 数据集:
    - Deep Learning Matrix Collection(DLMC)的稀疏矩阵, 稀疏度范围从50%到90%.
    - 通过应用各种剪枝技术到Transformer和ResNet-50模型中生成的2587个非结构化稀疏矩阵.
        - Transformer稀疏矩阵的行数(M)从512到33288不等
        - RenNet-50稀疏矩阵的行数(M)从64到2048不等
    - SuiteSparse矩阵集合的不同大小的稀疏矩阵, 主要包含稀疏度超过90%的稀疏矩阵

### 实验设置

### 性能评估

---

## 结论

由于非零元素的不规则分布和内存访问模式, 使用非结构化稀疏矩阵进行SpMM具有挑战性.
论文中, 开发了一种新颖的重排序算法, 通过在行聚类过程中考虑块稀疏模式来增强SpMM的数据局部性.

论文链接: [Accelerated Block-Sparsity-Aware Matrix Reordering for Leveraging Tensor Cores in Sparse Matrix-Multivector Multiplication](https://link.springer.com/chapter/10.1007/978-3-031-69583-4_1)

---