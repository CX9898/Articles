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
mkdir -p data
tar -zxvf lra_release.gz -C data
cd data
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar zxvf lra-release.gz 
mkdir lra_processed
cd ../src
python preprocess/create_pathfinder.py
python preprocess/create_listops.py
python preprocess/create_retrieval.py
python preprocess/create_text.py
python preprocess/create_cifar10.py
```

## 构建模型

[SAT](https://github.com/mlsys-lab-sogang/SAT?tab=readme-ov-file)

```shell
# 安装虚拟环境
sudo apt install python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
# 安装依赖
pip install tqdm
pip install numpy
pip install torch
pip install tensorboard
pip install requests
sh compile.sh
python main_learnable_skewness.py --mode train --task lra-image --random 1001 --name sat --sk 1.7 --ds 1.3

```
