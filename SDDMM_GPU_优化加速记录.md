# SDDMM_GPU 优化加速记录

---

## 加入 Tensor core

使用 Tensor core 直接对密集矩阵乘法进行计算, 计算时间

### 过程中发现的问题或优化点

#### 1:

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

#### 2:

核函数中加入了 `if()` 判断语句, 就算永远执行单一分支也比不加入 `if()` 判断语句时耗时久

加入判断语句(用时: 31.388 ms):

```c++
const int ldp = N;
const auto pOffsetPtr = matrixP + pRowId * ldp + pColId;
const float sparsity = calculateMatrixTileSparsity(WMMA_M, WMMA_N, ldp, MatrixStorageOrder::row_major, pOffsetPtr);
if (sparsity < 0) { // Always false

} else {
matrixTileMultiplicationUseTensorCore(pRowId, pColId, M, N, K, matrixA, matrixB, matrixS, matrixP);
}
```

不加入判断语句(用时: 21.8441 ms):

```c++
const int ldp = N;
const auto pOffsetPtr = matrixP + pRowId * ldp + pColId;
const float sparsity = calculateMatrixTileSparsity(WMMA_M, WMMA_N, ldp, MatrixStorageOrder::row_major, pOffsetPtr);
//    if (sparsity < 0) { // Always false

//    } else {
matrixTileMultiplicationUseTensorCore(pRowId, pColId, M, N, K, matrixA, matrixB, matrixS, matrixP);
//    }
```

#### 3:

blockDim 的尺寸改变的话时间大幅度降低.

原来的尺寸是 `block.x = 32` `block.y = 32`

```
M : 37000, N : 37000, K : 256, nnz : 368000, sparsity : 99.9731%
openTensorCoreMode
openTensorCoreMode matrixA : row = 37008, col = 256
openTensorCoreMode matrixB : row = 256, col = 37008
openTensorCoreMode matrixS : row = 37008, col = 37008
grid : [2313 73 1] block : [32 32 1]
Func comp_sddmm_gpu time : 23.8314 ms
```

改成了 `block.x = 128` `block.y = 4`

```
M : 37000, N : 37000, K : 256, nnz : 368000, sparsity : 99.9731%
openTensorCoreMode
openTensorCoreMode matrixA : row = 37008, col = 256
openTensorCoreMode matrixB : row = 256, col = 37008
openTensorCoreMode matrixS : row = 37008, col = 37008
grid : [579 579 1] block : [128 4 1]
Func comp_sddmm_gpu time : 13.0769 ms
```

从 grid 的数量上看, 改成128和4时, grid 数量大大增加了, 原先 2313 × 73 = 168849, 变成 579 × 579 = 335241

---

### 最初版本 测试结果

#### 测试结果 Debug build, 所有矩阵都是行主序储存

- GPU : 4090
- Debug build
- matrixA(half) : row_major, matrixB(half) : row_major, matrixP(float) : row_major
- WMMA : 16 × 16 × 16

| GPU : 4090, Debug build                                              | sddmm_isratnisa |  sddmm_zcx  |
|:---------------------------------------------------------------------|:---------------:|:-----------:|
| M : 1504, N : 1504, K : 256,<br/>nnz : 746316, sparsity : 67.0066%   |   2.13123 ms    | 0.059168 ms |
| M : 12432, N : 12432, K : 256,<br/>nnz : 746316, sparsity : 99.5171% |   0.419488 ms   | 3.06074 ms  |
| M : 8000, N : 8000, K : 256,<br/>nnz : 640000, sparsity : 99%        |   0.488832 ms   |  1.2759 ms  |
| M : 8000, N : 8000, K = 256,<br/>nnz: 1280000, sparsity : 98%        |   0.817376 ms   |    위에 같다    |
| M : 8000, N : 8000, K : 256,<br/>nnz : 1632000, sparsity : 97.45%    |   0.998016 ms   |    위에 같다    |
| M : 8000, N : 8000, K : 256,<br/>nnz : 1920000, sparsity : 97%       |   1.14563 ms    |    위에 같다    |
| M : 8000, N : 8000, K : 256,<br/>nnz : 2240000, sparsity : 96.5%     |   1.30128 ms    |    위에 같다    |
| M : 8000, N : 8000, K : 256,<br/> nnz : 2560000, sparsity : 96%      |   1.47882 ms    |    위에 같다    |
| M : 8000, N : 8000, K : 256,<br/>nnz : 6400000, sparsity : 90%       |   5.10989 ms    |    위에 같다    |

#### Release build

- Debug build
- GPU : 4090
- matrixA(half) : row_major, matrixB(half) : row_major, matrixP(float) : row_major

| 4090 |  |  |
|------|--|--|
|      |  |  |
|      |  |  |
|      |  |  |

---

### 使计算支持各个尺寸的矩阵(Matrix::openTensorCoreMode())

#### 测试结果 行主序储存 16×16×16

- GPU : 4090
- Debug build
- matrixA(half) : row_major, matrixB(half) : row_major, matrixP(float) : row_major
- WMMA : 16 × 16 × 16

| GPU:4090, debug build, row_major,16×16×16                            | sddmm_isratnisa | sddmm_zcx   |
|----------------------------------------------------------------------|-----------------|-------------|
| M : 3000, N : 7000, K : 256, nnz : 313110, sparsity : 98.51%         | 0.508448 ms     | 0.415744 ms |
| M : 2000, N : 12000, K : 256, nnz : 746000, sparsity : 96.8917%      | 1.59091 ms      | 0.448512 ms |
| M : 300000, N : 103000, K : 256, nnz : 69000000, sparsity : 99.7767% | 26.6559 ms      |             |
| M : 35000, N : 35000, K : 256, nnz : 422000, sparsity : 99.9656%     | 0.237888 ms     | 21.9749 ms  |
| M : 549000, N : 549000, K : 256, nnz : 926000, sparsity : 99.9997%   | 2.61411 ms      |             |
| M : 426000, N : 426000, K : 256, nnz : 1000000, sparsity : 99.9995%  | 2.41718 ms      |             |
| M : 37000, N : 37000, K : 256, nnz : 368000, sparsity : 99.9731%     | 0.234112 ms     | 23.9791 ms  |
| M : 4000, N : 4000, K : 256, nnz : 88000, sparsity : 99.45%          | 0.172064 ms     | 0.317472 ms |
| M : 106000, N : 106000, K : 256, nnz : 3000000, sparsity : 99.9733%  | 1.97805 ms      |             |
| M : 685000, N : 685000, K : 256, nnz : 8000000, sparsity : 99.9983%  | 12.3796 ms      |             |
| M : 916000, N : 916000, K : 256, nnz : 5000000, sparsity : 99.9994%  | 9.60342 ms      |             |
| M : 326000, N : 326000, K : 256, nnz : 1000000, sparsity : 99.9991%  | 2.11952 ms      |             |
| M : 197000, N : 197000, K : 256, nnz : 2000000, sparsity : 99.9948%  | 2.10243 ms      |             |
| M : 390000, N : 390000, K : 256, nnz : 2000000, sparsity : 99.9987%  | 3.45181 ms      |             |
| M : 260000, N : 260000, K : 256, nnz : 4000000, sparsity : 99.9941%  | 4.01792 ms      |             |
| M : 241000, N : 241000, K : 256, nnz : 561000, sparsity : 99.999%    | 1.28944 ms      |             |
| M : 36000, N : 36000, K : 256, nnz : 4000000, sparsity : 99.6914%    | 1.33123 ms      | 23.1115 ms  |

由于将稀疏矩阵按照0也储存的方式储存, 导致矩阵太大的情况下内存分配错误, 使得计算失败

#### 测试结果 行主序储存 32×8×16

- GPU : 4090
- Debug build
- matrixA(half) : row_major, matrixB(half) : row_major, matrixP(float) : row_major
- WMMA : 32×8×16

| GPU:4090, debug build, row_major, 32×8×16                            | sddmm_isratnisa | sddmm_zcx   |
|----------------------------------------------------------------------|-----------------|-------------|
| M : 3000, N : 7000, K : 256, nnz : 313110, sparsity : 98.51%         | 0.508448 ms     | 0.372704 ms |
| M : 2000, N : 12000, K : 256, nnz : 746000, sparsity : 96.8917%      | 1.59091 ms      | 0.44336 ms  |
| M : 300000, N : 103000, K : 256, nnz : 69000000, sparsity : 99.7767% | 26.6559 ms      |             |
| M : 35000, N : 35000, K : 256, nnz : 422000, sparsity : 99.9656%     | 0.237888 ms     | 17.8151 ms  |
| M : 549000, N : 549000, K : 256, nnz : 926000, sparsity : 99.9997%   | 2.61411 ms      |             |
| M : 426000, N : 426000, K : 256, nnz : 1000000, sparsity : 99.9995%  | 2.41718 ms      |             |
| M : 37000, N : 37000, K : 256, nnz : 368000, sparsity : 99.9731%     | 0.234112 ms     | 19.9316 ms  |
| M : 4000, N : 4000, K : 256, nnz : 88000, sparsity : 99.45%          | 0.172064 ms     | 0.317472 ms |
| M : 106000, N : 106000, K : 256, nnz : 3000000, sparsity : 99.9733%  | 1.97805 ms      |             |
| M : 685000, N : 685000, K : 256, nnz : 8000000, sparsity : 99.9983%  | 12.3796 ms      |             |
| M : 916000, N : 916000, K : 256, nnz : 5000000, sparsity : 99.9994%  | 9.60342 ms      |             |
| M : 326000, N : 326000, K : 256, nnz : 1000000, sparsity : 99.9991%  | 2.11952 ms      |             |
| M : 197000, N : 197000, K : 256, nnz : 2000000, sparsity : 99.9948%  | 2.10243 ms      |             |
| M : 390000, N : 390000, K : 256, nnz : 2000000, sparsity : 99.9987%  | 3.45181 ms      |             |
| M : 260000, N : 260000, K : 256, nnz : 4000000, sparsity : 99.9941%  | 4.01792 ms      |             |
| M : 241000, N : 241000, K : 256, nnz : 561000, sparsity : 99.999%    | 1.28944 ms      |             |
| M : 36000, N : 36000, K : 256, nnz : 4000000, sparsity : 99.6914%    | 1.33123 ms      | 19.9169 ms  |

由于将稀疏矩阵按照0也储存的方式储存, 导致矩阵太大的情况下内存分配错误, 使得计算失败

使用 32×8×16 的维度 比 16×16×16 更快了

#### 测试结果 行主序储存 8×32×16

- GPU : 4090
- Debug build
- matrixA(half) : row_major, matrixB(half) : row_major, matrixP(float) : row_major
- WMMA : 8×32×16

| GPU:4090, debug build, row_major, 8×32×16                            | sddmm_isratnisa | sddmm_zcx   |
|----------------------------------------------------------------------|-----------------|-------------|
| M : 3000, N : 7000, K : 256, nnz : 313110, sparsity : 98.51%         | 0.508448 ms     | 0.567232 ms |
| M : 2000, N : 12000, K : 256, nnz : 746000, sparsity : 96.8917%      | 1.59091 ms      | 0.577088 ms |
| M : 300000, N : 103000, K : 256, nnz : 69000000, sparsity : 99.7767% | 26.6559 ms      |             |
| M : 35000, N : 35000, K : 256, nnz : 422000, sparsity : 99.9656%     | 0.237888 ms     | 30.5323 ms  |
| M : 549000, N : 549000, K : 256, nnz : 926000, sparsity : 99.9997%   | 2.61411 ms      |             |
| M : 426000, N : 426000, K : 256, nnz : 1000000, sparsity : 99.9995%  | 2.41718 ms      |             |
| M : 37000, N : 37000, K : 256, nnz : 368000, sparsity : 99.9731%     | 0.234112 ms     | 34.361 ms   |
| M : 4000, N : 4000, K : 256, nnz : 88000, sparsity : 99.45%          | 0.172064 ms     | 0.398336 ms |
| M : 106000, N : 106000, K : 256, nnz : 3000000, sparsity : 99.9733%  | 1.97805 ms      |             |
| M : 685000, N : 685000, K : 256, nnz : 8000000, sparsity : 99.9983%  | 12.3796 ms      |             |
| M : 916000, N : 916000, K : 256, nnz : 5000000, sparsity : 99.9994%  | 9.60342 ms      |             |
| M : 326000, N : 326000, K : 256, nnz : 1000000, sparsity : 99.9991%  | 2.11952 ms      |             |
| M : 197000, N : 197000, K : 256, nnz : 2000000, sparsity : 99.9948%  | 2.10243 ms      |             |
| M : 390000, N : 390000, K : 256, nnz : 2000000, sparsity : 99.9987%  | 3.45181 ms      |             |
| M : 260000, N : 260000, K : 256, nnz : 4000000, sparsity : 99.9941%  | 4.01792 ms      |             |
| M : 241000, N : 241000, K : 256, nnz : 561000, sparsity : 99.999%    | 1.28944 ms      |             |
| M : 36000, N : 36000, K : 256, nnz : 4000000, sparsity : 99.6914%    | 1.33123 ms      | 32.917 ms   |

由于将稀疏矩阵按照0也储存的方式储存, 导致矩阵太大的情况下内存分配错误, 使得计算失败

使用 8×32×16 的维度 比 16×16×16 和 32×8×16 更慢了

---

### 在上个版本的基础上将 blockDIm 的尺寸改变成 128×4

blockDim 的尺寸改变的话时间大幅度降低.

#### 测试结果 行主序储存 16×16×16

- GPU : 4090
- Debug build
- matrixA(half) : row_major, matrixB(half) : row_major, matrixP(float) : row_major
- WMMA : 16 × 16 × 16

| GPU:4090, debug build, row_major,16×16×16                            | sddmm_isratnisa | sddmm_zcx   |
|----------------------------------------------------------------------|-----------------|-------------|
| M : 3000, N : 7000, K : 256, nnz : 313110, sparsity : 98.51%         | 0.508448 ms     | 0.280928 ms |
| M : 2000, N : 12000, K : 256, nnz : 746000, sparsity : 96.8917%      | 1.59091 ms      | 0.294144 ms |
| M : 300000, N : 103000, K : 256, nnz : 69000000, sparsity : 99.7767% | 26.6559 ms      |             |
| M : 35000, N : 35000, K : 256, nnz : 422000, sparsity : 99.9656%     | 0.237888 ms     | 16.2021 ms  |
| M : 549000, N : 549000, K : 256, nnz : 926000, sparsity : 99.9997%   | 2.61411 ms      |             |
| M : 426000, N : 426000, K : 256, nnz : 1000000, sparsity : 99.9995%  | 2.41718 ms      |             |
| M : 37000, N : 37000, K : 256, nnz : 368000, sparsity : 99.9731%     | 0.234112 ms     | 13.0718 ms  |
| M : 4000, N : 4000, K : 256, nnz : 88000, sparsity : 99.45%          | 0.172064 ms     | 0.189664 ms |
| M : 106000, N : 106000, K : 256, nnz : 3000000, sparsity : 99.9733%  | 1.97805 ms      |             |
| M : 685000, N : 685000, K : 256, nnz : 8000000, sparsity : 99.9983%  | 12.3796 ms      |             |
| M : 916000, N : 916000, K : 256, nnz : 5000000, sparsity : 99.9994%  | 9.60342 ms      |             |
| M : 326000, N : 326000, K : 256, nnz : 1000000, sparsity : 99.9991%  | 2.11952 ms      |             |
| M : 197000, N : 197000, K : 256, nnz : 2000000, sparsity : 99.9948%  | 2.10243 ms      |             |
| M : 390000, N : 390000, K : 256, nnz : 2000000, sparsity : 99.9987%  | 3.45181 ms      |             |
| M : 260000, N : 260000, K : 256, nnz : 4000000, sparsity : 99.9941%  | 4.01792 ms      |             |
| M : 241000, N : 241000, K : 256, nnz : 561000, sparsity : 99.999%    | 1.28944 ms      |             |
| M : 36000, N : 36000, K : 256, nnz : 4000000, sparsity : 99.6914%    | 1.33123 ms      | 14.1067 ms  |

由于将稀疏矩阵按照0也储存的方式储存, 导致矩阵太大的情况下内存分配错误, 使得计算失败

#### 测试结果 行主序储存 32×8×16

- GPU : 4090
- Debug build
- matrixA(half) : row_major, matrixB(half) : row_major, matrixP(float) : row_major
- WMMA : 32×8×16

| GPU:4090, debug build, row_major, 32×8×16                            | sddmm_isratnisa | sddmm_zcx   |
|----------------------------------------------------------------------|-----------------|-------------|
| M : 3000, N : 7000, K : 256, nnz : 313110, sparsity : 98.51%         | 0.508448 ms     | 0.33904 ms  |
| M : 2000, N : 12000, K : 256, nnz : 746000, sparsity : 96.8917%      | 1.59091 ms      | 0.417888 ms |
| M : 300000, N : 103000, K : 256, nnz : 69000000, sparsity : 99.7767% | 26.6559 ms      |             |
| M : 35000, N : 35000, K : 256, nnz : 422000, sparsity : 99.9656%     | 0.237888 ms     | 16.5034 ms  |
| M : 549000, N : 549000, K : 256, nnz : 926000, sparsity : 99.9997%   | 2.61411 ms      |             |
| M : 426000, N : 426000, K : 256, nnz : 1000000, sparsity : 99.9995%  | 2.41718 ms      |             |
| M : 37000, N : 37000, K : 256, nnz : 368000, sparsity : 99.9731%     | 0.234112 ms     | 18.4755 ms  |
| M : 4000, N : 4000, K : 256, nnz : 88000, sparsity : 99.45%          | 0.172064 ms     | 0.2536 ms   |
| M : 106000, N : 106000, K : 256, nnz : 3000000, sparsity : 99.9733%  | 1.97805 ms      |             |
| M : 685000, N : 685000, K : 256, nnz : 8000000, sparsity : 99.9983%  | 12.3796 ms      |             |
| M : 916000, N : 916000, K : 256, nnz : 5000000, sparsity : 99.9994%  | 9.60342 ms      |             |
| M : 326000, N : 326000, K : 256, nnz : 1000000, sparsity : 99.9991%  | 2.11952 ms      |             |
| M : 197000, N : 197000, K : 256, nnz : 2000000, sparsity : 99.9948%  | 2.10243 ms      |             |
| M : 390000, N : 390000, K : 256, nnz : 2000000, sparsity : 99.9987%  | 3.45181 ms      |             |
| M : 260000, N : 260000, K : 256, nnz : 4000000, sparsity : 99.9941%  | 4.01792 ms      |             |
| M : 241000, N : 241000, K : 256, nnz : 561000, sparsity : 99.999%    | 1.28944 ms      |             |
| M : 36000, N : 36000, K : 256, nnz : 4000000, sparsity : 99.6914%    | 1.33123 ms      | 18.8892 ms  |

由于将稀疏矩阵按照0也储存的方式储存, 导致矩阵太大的情况下内存分配错误, 使得计算失败

blockDim使用128×8的情况下, 使用 32×8×16 的维度 比 16×16×16 更慢了

#### 测试结果 行主序储存 8×32×16

- GPU : 4090
- Debug build
- matrixA(half) : row_major, matrixB(half) : row_major, matrixP(float) : row_major
- WMMA : 8×32×16

| GPU:4090, debug build, row_major, 8×32×16                            | sddmm_isratnisa | sddmm_zcx   |
|----------------------------------------------------------------------|-----------------|-------------|
| M : 3000, N : 7000, K : 256, nnz : 313110, sparsity : 98.51%         | 0.508448 ms     | 0.311552 ms |
| M : 2000, N : 12000, K : 256, nnz : 746000, sparsity : 96.8917%      | 1.59091 ms      | 0.326208 ms |
| M : 300000, N : 103000, K : 256, nnz : 69000000, sparsity : 99.7767% | 26.6559 ms      |             |
| M : 35000, N : 35000, K : 256, nnz : 422000, sparsity : 99.9656%     | 0.237888 ms     | 20.6243 ms  |
| M : 549000, N : 549000, K : 256, nnz : 926000, sparsity : 99.9997%   | 2.61411 ms      |             |
| M : 426000, N : 426000, K : 256, nnz : 1000000, sparsity : 99.9995%  | 2.41718 ms      |             |
| M : 37000, N : 37000, K : 256, nnz : 368000, sparsity : 99.9731%     | 0.234112 ms     | 16.8907 ms  |
| M : 4000, N : 4000, K : 256, nnz : 88000, sparsity : 99.45%          | 0.172064 ms     | 0.214592 ms |
| M : 106000, N : 106000, K : 256, nnz : 3000000, sparsity : 99.9733%  | 1.97805 ms      |             |
| M : 685000, N : 685000, K : 256, nnz : 8000000, sparsity : 99.9983%  | 12.3796 ms      |             |
| M : 916000, N : 916000, K : 256, nnz : 5000000, sparsity : 99.9994%  | 9.60342 ms      |             |
| M : 326000, N : 326000, K : 256, nnz : 1000000, sparsity : 99.9991%  | 2.11952 ms      |             |
| M : 197000, N : 197000, K : 256, nnz : 2000000, sparsity : 99.9948%  | 2.10243 ms      |             |
| M : 390000, N : 390000, K : 256, nnz : 2000000, sparsity : 99.9987%  | 3.45181 ms      |             |
| M : 260000, N : 260000, K : 256, nnz : 4000000, sparsity : 99.9941%  | 4.01792 ms      |             |
| M : 241000, N : 241000, K : 256, nnz : 561000, sparsity : 99.999%    | 1.28944 ms      |             |
| M : 36000, N : 36000, K : 256, nnz : 4000000, sparsity : 99.6914%    | 1.33123 ms      | 16.0298 ms  |

由于将稀疏矩阵按照0也储存的方式储存, 导致矩阵太大的情况下内存分配错误, 使得计算失败

blockDim使用128×8的情况下, 使用 8×32×16 的维度有的变快有的变慢