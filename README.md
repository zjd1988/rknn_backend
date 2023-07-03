# rknn_backend
## 编译步骤
### 1 拉取triton-server仓库 
```
# 以我本地的目录和环境为例，以下都是在3588开发板进行，交叉编译暂时没尝试

cd /data/github_codes
git clone https://github.com/triton-inference-server/server.git
```

### 2 切换到server目录，执行python ./build.py 生成cmake_build编译脚本
```
#生成不需要docker镜像的编译脚本,--build-dir 使用完整路径(以/data/github_codes/server/build_test为例)
#注意需要使能enable-mali-gpu，否则无法支持npu多实例

cd /data/github_codes/server
python ./build.py -v --dryrun --no-container-build --backend=ensemble \
--backend=python --endpoint=grpc --endpoint=http --enable-logging \
--enable-stats --enable-metrics --enable-cpu-metrics --enable-tracing \
--enable-mali-gpu --build-dir=/data/github_codes/server/build_test
```

### 3 拉取rknn_backend仓库到build_test路径下
```
cd build_test
git clone https://github.com/zjd1988/rknn_backend.git rknn
```

### 4 修改cmake_build，添加rknn_backend相关脚本命令
```
# 执行完成第一步会在/data/github_codes/server/build_test目录下生成cmake_build
# 在该文件最后添加下面的rknn_backend相关编译脚本命令

########
# 'rknn' backend
# Delete this section to remove backend from build
#
mkdir -p /data/github_codes/server/build_test
cd /data/github_codes/server/build_test
#rm -fr rknn

mkdir -p /data/github_codes/server/build_test/rknn/build
cd /data/github_codes/server/build_test/rknn/build
cmake \
    "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" \
    "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DTRITON_BUILD_CONTAINER_VERSION=23.05" \
    "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/data/github_codes/server/build_test/rknn/install" \
    "-DTRITON_COMMON_REPO_TAG:STRING=r23.05" "-DTRITON_CORE_REPO_TAG:STRING=r23.05" \
    "-DTRITON_BACKEND_REPO_TAG:STRING=r23.05" "-DTRITON_ENABLE_GPU:BOOL=OFF" \
    "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" \
    "-DTRITON_ENABLE_METRICS:BOOL=ON" ..
#make -j16 VERBOSE=1 install
make -j install
mkdir -p /data/github_codes/server/build_test/opt/tritonserver/backends
rm -fr /data/github_codes/server/build_test/opt/tritonserver/backends/rknn
cp -r /data/github_codes/server/build_test/rknn/install/backends/rknn /data/github_codes/server/build_test/opt/tritonserver/backends
#
# end 'rknn' backend
########
```
### 5 完整的编译脚本如下(option)
```
#!/usr/bin/env bash

#
# Build script for Triton Inference Server
#

# Exit script immediately if any command fails
set -e
set -x

########
# Triton core library and tritonserver executable
#
mkdir -p /data/github_codes/server/build_test/tritonserver/build
cd /data/github_codes/server/build_test/tritonserver/build
cmake \
    "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" \
    "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DCMAKE_BUILD_TYPE=Release" \
    "-DCMAKE_INSTALL_PREFIX:PATH=/data/github_codes/server/build_test/tritonserver/install" \
    "-DTRITON_VERSION:STRING=2.34.0" "-DTRITON_COMMON_REPO_TAG:STRING=r23.05" \
    "-DTRITON_CORE_REPO_TAG:STRING=r23.05" "-DTRITON_BACKEND_REPO_TAG:STRING=r23.05" \
    "-DTRITON_THIRD_PARTY_REPO_TAG:STRING=r23.05" "-DTRITON_ENABLE_LOGGING:BOOL=ON" \
    "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" \
    "-DTRITON_ENABLE_METRICS_GPU:BOOL=OFF" "-DTRITON_ENABLE_METRICS_CPU:BOOL=ON" \
    "-DTRITON_ENABLE_TRACING:BOOL=ON" "-DTRITON_ENABLE_NVTX:BOOL=OFF" \
    "-DTRITON_ENABLE_GPU:BOOL=OFF" \
    "-DTRITON_MIN_COMPUTE_CAPABILITY=6.0" "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" \
    "-DTRITON_ENABLE_GRPC:BOOL=ON" "-DTRITON_ENABLE_HTTP:BOOL=ON" \
    "-DTRITON_ENABLE_SAGEMAKER:BOOL=OFF" "-DTRITON_ENABLE_VERTEX_AI:BOOL=OFF" \
    "-DTRITON_ENABLE_GCS:BOOL=OFF" "-DTRITON_ENABLE_S3:BOOL=OFF" \
    "-DTRITON_ENABLE_AZURE_STORAGE:BOOL=OFF" "-DTRITON_ENABLE_ENSEMBLE:BOOL=ON" \
    "-DTRITON_ENABLE_TENSORRT:BOOL=OFF" /data/github_codes/server
#make -j16 VERBOSE=1 install
make -j16 install
mkdir -p /data/github_codes/server/build_test/opt/tritonserver/bin
cp /data/github_codes/server/build_test/tritonserver/install/bin/tritonserver /data/github_codes/server/build_test/opt/tritonserver/bin
mkdir -p /data/github_codes/server/build_test/opt/tritonserver/lib
cp /data/github_codes/server/build_test/tritonserver/install/lib/libtritonserver.so /data/github_codes/server/build_test/opt/tritonserver/lib
mkdir -p /data/github_codes/server/build_test/opt/tritonserver/include/triton
cp -r /data/github_codes/server/build_test/tritonserver/install/include/triton/core /data/github_codes/server/build_test/opt/tritonserver/include/triton/core
cp /data/github_codes/server/LICENSE /data/github_codes/server/build_test/opt/tritonserver
cp /data/github_codes/server/TRITON_VERSION /data/github_codes/server/build_test/opt/tritonserver
#
# end Triton core library and tritonserver executable
########

########
# 'python' backend
# Delete this section to remove backend from build
#
mkdir -p /data/github_codes/server/build_test
cd /data/github_codes/server/build_test
rm -fr python
if [[ ! -e python ]]; then
  git clone --recursive --single-branch --depth=1 -b r23.05 https://github.com/triton-inference-server/python_backend.git python;
fi
mkdir -p /data/github_codes/server/build_test/python/build
cd /data/github_codes/server/build_test/python/build
cmake "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/data/github_codes/server/build_test/python/install" "-DTRITON_COMMON_REPO_TAG:STRING=r23.05" "-DTRITON_CORE_REPO_TAG:STRING=r23.05" "-DTRITON_BACKEND_REPO_TAG:STRING=r23.05" "-DTRITON_ENABLE_GPU:BOOL=OFF" "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" ..
make -j16 VERBOSE=1 install
mkdir -p /data/github_codes/server/build_test/opt/tritonserver/backends
rm -fr /data/github_codes/server/build_test/opt/tritonserver/backends/python
cp -r /data/github_codes/server/build_test/python/install/backends/python /data/github_codes/server/build_test/opt/tritonserver/backends
#
# end 'python' backend
########

########
# 'rknn' backend
# Delete this section to remove backend from build
#
mkdir -p /data/github_codes/server/build_test
cd /data/github_codes/server/build_test
#rm -fr rknn

mkdir -p /data/github_codes/server/build_test/rknn/build
cd /data/github_codes/server/build_test/rknn/build
cmake \
    "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" \
    "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DTRITON_BUILD_CONTAINER_VERSION=23.05" \
    "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/data/github_codes/server/build_test/rknn/install" \
    "-DTRITON_COMMON_REPO_TAG:STRING=r23.05" "-DTRITON_CORE_REPO_TAG:STRING=r23.05" \
    "-DTRITON_BACKEND_REPO_TAG:STRING=r23.05" "-DTRITON_ENABLE_GPU:BOOL=OFF" \
    "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" \
    "-DTRITON_ENABLE_METRICS:BOOL=ON" ..
#make -j1 VERBOSE=1 install
make -j install
mkdir -p /data/github_codes/server/build_test/opt/tritonserver/backends
rm -fr /data/github_codes/server/build_test/opt/tritonserver/backends/rknn
cp -r /data/github_codes/server/build_test/rknn/install/backends/rknn /data/github_codes/server/build_test/opt/tritonserver/backends
#
# end 'rknn' backend
########
```
### 6 切换到server目录下，执行编译
```
cd ../
./build_test/cmake_build

# 初次编译耗时较长(1h+),最终会在build_test/opt/tritonserver目录下存放编译生成的所有内容
```

## 测试
### 1 启动triton-server服务
```
./build_test/opt/tritonserver/bin/tritonserver  --model-repository /data/github_codes/server/build_test/rknn/examples/models --backend-directory /data/github_codes/server/build_test/opt/tritonserver/backends/
```

### 2 yolov5模型测试
```
# 需要预先在python环境安装 tritonclient
python -m pip install tritonclient[all]
cd /data/github_codes/server/build_test/rknn/examples/yolov5
python test_yolov5.py

# 可以通过修改/data/github_codes/server/build_test/rknn/examples/models/yolov5/config.pbtxt
# 配置模型实例个数和模型加载的npu核id
# 注：目前，暂时不支持模型core_mask按RKNN_NPU_CORE_0_1和RKNN_NPU_CORE_0_1_2进行加载
```
![yolov5测试结果](https://github.com/zjd1988/rknn_backend/blob/main/examples/yolov5/yolov5_result.jpg)

### 3 single_input (mobilenet) 模型测试
```
cd /data/github_codes/server/build_test/rknn/examples/single_input
python test_single_input.py

# mobilenet test top3 result
# Shih-Tzu:0.984375
# Pekinese:0.0078125
# Lhasa:0.00390625
```

### 4 multi_input 模型测试
```
cd /data/github_codes/server/build_test/rknn/examples/multi_input
python test_multi_input.py

# multi input result shape
# (1, 8, 128, 128)
```

### 5 ensemble_mobilenet 模型测试(ensemble+python)
```
cd /data/github_codes/server/build_test/rknn/examples/ensemble_mobilenet
python test_ensemble_mobilenet.py --image ../single_input/dog_224x224.jpg --label_file ../single_input/labels.txt

# Result is class: Shih-Tzu
```
