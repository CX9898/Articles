`# Nsight Compute使用笔记

## 前言

一个小节详细讲一个功能. 一个小节详细讲怎么查找某一个问题.

---

## 什么是Nsight Compute

---

## 为什么要使用Nsight Compute

---

## Nsight Compute

### 启动Nsight Compute 输出指定核函数信息

---

### 判断 memory bound 还是 compute bound

GPU Speed of Light Throughput(GPU 光速吞吐率)是 用来衡量 CUDA kernel 相对于 GPU 理论峰值资源利用率的高低.

- Compute (SM) Throughput [%]

| 项目                            | 含义                                      | 解读方式                                           |
|-------------------------------|-----------------------------------------|------------------------------------------------|
| Compute (SM) Throughput (%)   | 当前 kernel 的计算单元（SM）吞吐率，占理论 FP32 峰值的百分比。 | 越接近 100%，说明计算资源利用越充分。低于 60% 通常代表计算能力没有被用满。     |
| Memory Throughput (%)         | 总内存（主要是 DRAM）带宽利用率，占理论带宽的百分比。           | 越高代表内存访问越密集。低于 60% 可能说明 memory access 存在延迟或空闲。 |
| L1/TEX Cache Throughput (%)   | L1 cache 或 texture cache 吞吐率。           | 可辅助分析 memory access pattern 的局部性（locality）。    |
| L2 Cache Throughput (%)       | L2 cache 吞吐率。                           | 如果很高，说明很多数据命中 L2，可减小 DRAM 压力。                  |
| DRAM Throughput (%)           | 全局内存带宽利用率（通常与 Memory Throughput 相同）。    | 高表示频繁访问 global memory，低可能说明内存未有效使用或隐藏了延迟。      |
| Duration (%)                  | Kernel 执行的总耗时。                          | 越短越好，是衡量优化是否生效的重要指标。                           |
| Elapsed Cycles                | 总共执行的时钟周期。                              | 可用于推断效率和 SM 的活跃度。                              |
| SM Active Cycles              | Streaming Multiprocessors 实际参与工作的周期。    | 越接近 Elapsed Cycles 越好，反映 SM 活跃程度。              |
| SM Frequency / DRAM Frequency | GPU 时钟频率（单位：cycles/ns）                  | 提供理论计算频率和内存频率参考。                               |

-

### 查看Bank-conflict

---


