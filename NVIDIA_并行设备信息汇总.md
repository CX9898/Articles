# Nvidia 并行设备架构信息汇总

前言

做并行计算避免不了跟硬件打交道, 熟悉硬件架构更有利于写出高效并行的代码.
这里用于记录 Nvidia GPU 架构来方便查看, 本文可能会不定时更新.

---

## 计算能力汇总

## 架构汇总

## GPU 详细信息

### Ada Lovelace 架构

NVIDIA GeForce RTX 4090

- GPU架构: Ada Lovelace
- 计算能力: 8.9
- 多处理器(SM)数量: 128
- CUDA核心数: 16384
- Tensor核心数: 512
- 显存位宽: 384位 
- Warp大小: 32
- 每个多处理器(SM)的最大线程数: 1536
- 每个多处理器(SM)的共享内存总量：102400 bytes
- 常量内存总量：65536 bytes
- 每个thread block的最大线程数: 1024
- Thread block的最大尺寸(x, y, z): (1024, 1024, 64)
- Grid的最大尺寸(x, y, z): (2147483647, 65535, 65535)
- 全局内存总量：24109 MB(25280184320 bytes)
- 每个thread block可用的共享内存大小: 49152 bytes
- 每个thread block可用的寄存器数量: 65536
- L2缓存大小: 75497472 bytes
- 纹理对齐：512 bytes

---