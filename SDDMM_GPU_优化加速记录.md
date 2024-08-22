# SDDMM_GPU 优化加速记录

---

## 加入 Tensor core

使用 Tensor core 直接对密集矩阵乘法进行计算, 计算时间

### 问题:

函数 `__shfl_xor` 在 cuda 后续版本已经弃用, 应该改为使用 `__shfl_xor_sync` 

但是发现使用 `__shfl_xor` 和 `__shfl_xor_sync` 的结果不一致

```c++
sm1 += __shfl_xor(sm1, 1); // 使用shuffle指令. 使线程0的sm1加到线程1的sm1上, 线程1的sm1加到线程2的sm2上
sm2 += __shfl_xor(sm2, 1);
```

```c++
sm1 += __shfl_xor_sync(0xFFFFFFFF, sm1, 1); // 使用shuffle指令. 使线程0的sm1加到线程1的sm1上, 线程1的sm1加到线程2的sm2上
sm2 += __shfl_xor_sync(0xFFFFFFFF, sm2, 1);
```

---

### 最初版本 测试结果

#### Debug mode

4090

| GPU : 4090, Debug build                                               | sddmm_isratnisa | sddmm_zcx   |
|:--------------------------------------------------------------------- |:---------------:|:-----------:|
| M : 1504, N : 1504, K : 256, <br/>nnz : 746316, sparsity : 67.0066%   | 2.13123 ms      | 0.059168 ms |
| M : 12432, N : 12432, K : 256, <br/>nnz : 746316, sparsity : 99.5171% | 0.419488 ms     | 3.06074 ms  |
| M : 8000, N : 8000, K : 256, <br/>nnz : 640000, sparsity : 99%        | 0.488832 ms     | 1.2759 ms   |
| M : 8000, N : 8000, K = 256, <br/>nnz: 1280000, sparsity : 98%        | 0.817376 ms     | 위에 같다       |
| M : 8000, N : 8000, K : 256, <br/>nnz : 1632000, sparsity : 97.45%    | 0.998016 ms     | 위에 같다       |
| M : 8000, N : 8000, K : 256, <br/>nnz : 1920000, sparsity : 97%       | 1.14563 ms      | 위에 같다       |
| M : 8000, N : 8000, K : 256, <br/>nnz : 2240000, sparsity : 96.5%     | 1.30128 ms      | 위에 같다       |
| M : 8000, N : 8000, K : 256,<br/> nnz : 2560000, sparsity : 96%       | 1.47882 ms      | 위에 같다       |
| M : 8000, N : 8000, K : 256, <br/>nnz : 6400000, sparsity : 90%       | 5.10989 ms      | 위에 같다       |

#### Release mode

| 4090 |     |     |
| ---- | --- | --- |
|      |     |     |
|      |     |     |
|      |     |     |



---

#### sddmm_isratnisa :

M : 1500, N : 12419, K = 256

time : 2.14611 ms

---

M : 1504, N : 1504, K = 256, nnz: 746316

Tilesize: X = 50000, tilesize: Y = 192, TB: 512

Time for SDDMM with K = 256 : 2.1273 ms

---

#### sddmm_zcx :

M : 12432, N : 12432, K : 256

time : 3.05766ms

---

M : 1504, N : 1504, K : 256, nnz : 746316
Func comp_sddmm_gpu time : 0.059392ms

Checking results...
Error : idx = 32937 data1 = 6791.168945, data2 = 6790.420410
Error : idx = 37682 data1 = 6621.742676, data2 = 6621.058105
Error : idx = 62951 data1 = 6490.582031, data2 = 6489.914062
Error : idx = 71769 data1 = 6745.477051, data2 = 6744.735352
Error : idx = 78805 data1 = 6967.206543, data2 = 6968.077637
Error : idx = 103173 data1 = 6405.111328, data2 = 6405.812988
Error : idx = 103569 data1 = 5965.936523, data2 = 5966.629395
Error : idx = 123167 data1 = 6707.535645, data2 = 6708.208496
Error : idx = 148660 data1 = 6723.708008, data2 = 6724.389160
Inconsistent data! 90 errors!



---

### 使计算支持各个尺寸的矩阵(Matrix::openTensorCoreMode())

| GPU:4090 debug compile                                            | sddmm_isratnisa | sddmm_zcx   |
| ----------------------------------------------------------------- | --------------- | ----------- |
| M : 3000, N : 7000, K : 256, <br/>nnz : 313110, sparsity : 98.51% | 0.508448 ms     | 0.415744 ms |
|                                                                   |                 |             |
|                                                                   |                 |             |
