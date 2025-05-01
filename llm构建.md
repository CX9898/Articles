# llm构建

## 构建 llama

安装依赖:

- CMake
- GCC
- CURL

安装CURL:

```shell
sudo apt-get update
sudo apt-get install libcurl4-openssl-dev
```

构建llama:

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## 数据准备阶段

[Skyformer](https://github.com/pkuzengqi/Skyformer?tab=readme-ov-file)

```shell
git clone https://github.com/pkuzengqi/Skyformer.git
cd Skyformer

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install 'tensorflow>=2.3.1' 'tensorflow-datasets>=4.0.1' 'tensorboard>=2.3.0'

wget https://storage.googleapis.com/long-range-arena/pathfinder_tfds.gz
gunzip pathfinder_tfds.gz
export _PATHFINDER_TFDS_PATH=~/CLionProjects/Skyformer/pathfinder_tfds
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
mkdir -p data
tar -zxvf lra_release.gz -C data
mkdir -p data/lra_processed

cd src
python preprocess/create_pathfinder.py
python preprocess/create_listops.py
python preprocess/create_retrieval.py
python preprocess/create_text.py
python preprocess/create_cifar10.py # 训练过程中出现内存分配失败. 限制训练样本数量为10000000后成功.
```

## 构建SAT模型

[SAT](https://github.com/mlsys-lab-sogang/SAT?tab=readme-ov-file)

```shell
# 安装虚拟环境

mkdir -p data
cp ../Skyformer/data/lra_processed/ data/ -r

sudo apt install python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
# 安装依赖
pip install tqdm
pip install numpy==1.26.4
pip install torch
pip install tensorboard
pip install requests
sh compile.sh
python main_learnable_skewness.py --mode train --task lra-image --random 1001 --name sat --sk 1.7 --ds 1.3 # 训练过程中出现GPU内存不足的问题. 减小batch size后成功.
python main_inference.py --mode eval --task lra-image --random 1001 --name sat
```

# 参数说明文档（SAT-Optimization）

本项目支持多个 LRA 任务（如 `lra-image`, `lra-retrieval` 等），模型结构、训练流程、稀疏切换逻辑均可配置。以下是命令行参数和配置文件参数的详细说明。

---

## 🖥️ 命令行参数

| 参数名             | 示例值             | 说明                                    |
|-----------------|-----------------|---------------------------------------|
| `--mode`        | `train`, `eval` | 运行模式：训练或评估                            |
| `--task`        | `lra-image`     | 指定任务名，对应配置文件中的任务键                     |
| `--random`      | `1001`          | 随机种子，用于复现训练和决定输出目录名                   |
| `--name`        | `sat`           | 模型命名前缀，决定 checkpoint 和 `mat_lst` 的保存名 |
| `--sk`          | `1.7`           | 稀疏切换判据：注意力 skewness 阈值                |
| `--ds`          | `1.3`           | 稀疏切换判据：注意力 Frobenius 距离变化阈值           |
| `--train_steps` | `5000`（可选）      | 可覆盖配置中的 `num_train_steps`             |

---

## 🧠 配置结构说明（`config.py`）

每个任务配置分为三块：`dataset`、`model`、`training`

---

### 📦 dataset 字段

| 参数      | 说明           |
|---------|--------------|
| `train` | 训练集样本数（统计信息） |
| `dev`   | 验证集样本数       |
| `test`  | 测试集样本数       |

---

### 🧠 model 字段（模型结构）

| 参数名                      | 示例值      | 说明                               |
|--------------------------|----------|----------------------------------|
| `learn_pos_emb`          | `True`   | 是否使用可学习的位置编码                     |
| `tied_weights`           | `False`  | 是否共享 encoder/decoder 权重          |
| `embedding_dim`          | `64`     | 输入嵌入维度                           |
| `transformer_dim`        | `64`     | transformer 层的通道数                |
| `transformer_hidden_dim` | `128`    | FFN 隐藏层维度                        |
| `head_dim`               | `32`     | 每个 attention head 的维度            |
| `num_head`               | `2`      | attention head 数量                |
| `num_layers`             | `2`      | transformer 层数                   |
| `vocab_size`             | `256`    | 输入词表大小                           |
| `max_seq_len`            | `1024`   | 最大输入序列长度                         |
| `dropout_prob`           | `0.1`    | Dropout 概率（非 attention）          |
| `attention_dropout`      | `0.1`    | Attention dropout 概率             |
| `pooling_mode`           | `"MEAN"` | 输出池化方式（`MEAN`, `CLS`）            |
| `num_classes`            | `10`     | 输出类别数（分类任务）                      |
| `block_size`             | `32`     | 稀疏 attention 中的 block 尺寸         |
| `batch_size`             | `256`    | 每张 GPU 的 mini-batch 大小           |
| `density`                | `0.05`   | 稀疏 attention 中的 block 稠密度（越小越稀疏） |

---

### 🏃 training 字段（训练流程）

| 参数名                     | 示例值        | 说明                                   |
|-------------------------|------------|--------------------------------------|
| `batch_size`            | `256`      | 总训练 batch size                       |
| `learning_rate`         | `0.002`    | 学习率                                  |
| `warmup`                | `175`      | warmup 步数（线性增长）                      |
| `lr_decay`              | `"linear"` | 学习率下降策略（如线性）                         |
| `weight_decay`          | `0`        | L2 正则项系数                             |
| `eval_frequency`        | `500`      | 每多少步进行一次验证                           |
| `num_train_steps`       | `10000`    | 总训练步数                                |
| `num_init_steps`        | `0`        | 初始化阶段步数（可忽略）                         |
| `num_eval_steps`        | `20`       | 每次验证时运行多少 batch                      |
| `num_dense_train_steps` | `10`       | dense attention 的训练步数上限              |
| `attn_loss_scale`       | `0.01`     | attention loss 的缩放因子                 |
| `skewness`              | `1.7`      | 稀疏切换判据：注意力分布偏度阈值                     |
| `distance`              | `1.3`      | 稀疏切换判据：attention 分布变化率（Frobenius 距离） |
| `patience`              | `10`       | 提前停止验证集准确度无提升的容忍次数                   |

---

## 🧬 稀疏 attention 自动切换机制

模型会在训练过程中检测以下两个指标：

- **skewness（偏度）**：注意力分布是否偏离平均
- **Frobenius distance**：当前与前一次注意力矩阵之间的变化量

当满足以下条件时会切换到 block-sparse attention：

```python
if skewness > sk_threshold and relative_distance_change > dist_threshold:
    transition = True
