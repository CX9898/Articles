![CUDA优化并行归约](img/CUDA优化并行归约/封面.png)

# CUDA优化并行归约

**[Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)**

NVIDIA官方很经典的一个文档, 介绍了如何在CUDA中实现并行归约, 并且一步一步地优化, 从而达到最优性能. 总共讲解了7种不同的优化版本.
本文是对这个文档的翻译, 并且加入了一些自己的理解和注释.

## 优化目标:

- 应该努力达到GPU的峰值性能
- 选择正确的度量标准:
    - GFLOP/s: 用于计算受限的内核
    - 带宽: 针对内存受限的内核
- 归约操作的计算强度非常低, 每个元素加载后只进行对应一次加法运算. 主要受限于内存带宽而不是计算能力
- 因此, 努力的方向应该是争取达到峰值带宽

## 1: Interleaved Addressing

Interleaved Addressing(交错寻址)是最简单的并行归约实现.

```c++
1 __global__ void reduce0(int *g_idata, int *g_odata) {
2     extern __shared__ int sdata[];
3 
4     unsigned int tid = threadIdx.x;
5     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
6     sdata[tid] = g_idata[i];
7     __syncthreads();
8 
9     for (unsigned int s = 1; s < blockDim.x; s *= 2) {
10         if (tid % (2 * s) == 0) {
11             sdata[tid] += sdata[tid + s];
12         }
13         __syncthreads();
14     }
15
16     if (tid == 0) g_odata[blockIdx.x] = sdata[0];
17 }
```

以上代码重点关注第9-14行的for循环, 这个循环是并行归约的核心部分. 每次循环, 每个线程都会将自己的数据与距离为s的线程的数据相加.
通过这种方式, 每次循环都会将数组的大小减半, 直到最后只剩下一个元素.
但在这种方式下, 由于第10行的if判断, 导致了线程的分支发散, 让warp非常低效的执行. 并且`%` 运算符的速度非常慢.

---

## 总结

参考:
[1] [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)