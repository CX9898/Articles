# llm构建

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