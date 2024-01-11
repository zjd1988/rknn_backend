## 1 拉取triton-server仓库和安装相关依赖
```
# 以我本地的硬件和环境为例(使用orange pi 5b, 镜像版本Orangepi5b_1.0.4_ubuntu_jammy_server_linux5.10.110.7z),
# 交叉编译暂时没尝试

cd /data/github_codes
git clone https://github.com/triton-inference-server/server.git
cd /data/github_codes/server/
cd server && git checkout r23.12

cd /data/github_codes/
wget https://jaist.dl.sourceforge.net/project/boost/boost/1.80.0/boost_1_80_0.tar.gz && \
    tar xzf boost_1_80_0.tar.gz && cd boost_1_80_0 && ./bootstrap.sh --prefix=/usr && \
    ./b2 install &&  mv /data/github_codes/boost_1_80_0/boost /usr/include/boost
```

## 2 切换到server目录，执行python ./build.py 生成cmake_build编译脚本
```
# 生成不需要docker镜像的编译脚本,--build-dir 使用完整路径(以/data/github_codes/server/build为例)
# 注意需要使能enable-mali-gpu，否则无法支持npu多实例

cd /data/github_codes/server
python ./build.py -v --dryrun --no-container-build --backend=ensemble \
--backend=python --backend=onnxruntime --endpoint=grpc --endpoint=http --enable-logging \
--enable-stats --enable-metrics --enable-cpu-metrics --enable-tracing \
--enable-mali-gpu --build-dir=$PWD/build

```

## 3 按照cmake_build， 依次编译不同模块, cat /data/github_codes/server/build/cmake_build 

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
mkdir -p /data/github_codes/server/build/tritonserver/build
cd /data/github_codes/server/build/tritonserver/build
cmake "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/data/github_codes/server/build/tritonserver/install" "-DTRITON_VERSION:STRING=2.41.0" "-DTRITON_COMMON_REPO_TAG:STRING=r23.12" "-DTRITON_CORE_REPO_TAG:STRING=r23.12" "-DTRITON_BACKEND_REPO_TAG:STRING=r23.12" "-DTRITON_THIRD_PARTY_REPO_TAG:STRING=r23.12" "-DTRITON_ENABLE_LOGGING:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" "-DTRITON_ENABLE_METRICS_GPU:BOOL=OFF" "-DTRITON_ENABLE_METRICS_CPU:BOOL=ON" "-DTRITON_ENABLE_TRACING:BOOL=ON" "-DTRITON_ENABLE_NVTX:BOOL=OFF" "-DTRITON_ENABLE_GPU:BOOL=OFF" "-DTRITON_MIN_COMPUTE_CAPABILITY=6.0" "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_GRPC:BOOL=ON" "-DTRITON_ENABLE_HTTP:BOOL=ON" "-DTRITON_ENABLE_SAGEMAKER:BOOL=OFF" "-DTRITON_ENABLE_VERTEX_AI:BOOL=OFF" "-DTRITON_ENABLE_GCS:BOOL=OFF" "-DTRITON_ENABLE_S3:BOOL=OFF" "-DTRITON_ENABLE_AZURE_STORAGE:BOOL=OFF" "-DTRITON_ENABLE_ENSEMBLE:BOOL=ON" "-DTRITON_ENABLE_TENSORRT:BOOL=OFF" /data/github_codes/server
make -j16 VERBOSE=1 install
mkdir -p /data/github_codes/server/build/opt/tritonserver/bin
cp /data/github_codes/server/build/tritonserver/install/bin/tritonserver /data/github_codes/server/build/opt/tritonserver/bin
mkdir -p /data/github_codes/server/build/opt/tritonserver/lib
cp /data/github_codes/server/build/tritonserver/install/lib/libtritonserver.so /data/github_codes/server/build/opt/tritonserver/lib
mkdir -p /data/github_codes/server/build/opt/tritonserver/python
cp /data/github_codes/server/build/tritonserver/install/python/tritonserver*.whl /data/github_codes/server/build/opt/tritonserver/python
mkdir -p /data/github_codes/server/build/opt/tritonserver/include/triton
cp -r /data/github_codes/server/build/tritonserver/install/include/triton/core /data/github_codes/server/build/opt/tritonserver/include/triton/core
cp /data/github_codes/server/LICENSE /data/github_codes/server/build/opt/tritonserver
cp /data/github_codes/server/TRITON_VERSION /data/github_codes/server/build/opt/tritonserver
#
# end Triton core library and tritonserver executable
########

########
# 'python' backend
# Delete this section to remove backend from build
#
mkdir -p /data/github_codes/server/build
cd /data/github_codes/server/build
rm -fr python
if [[ ! -e python ]]; then
  git clone --recursive --single-branch --depth=1 -b r23.12 https://github.com/triton-inference-server/python_backend.git python;
fi
mkdir -p /data/github_codes/server/build/python/build
cd /data/github_codes/server/build/python/build
cmake "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/data/github_codes/server/build/python/install" "-DTRITON_COMMON_REPO_TAG:STRING=r23.12" "-DTRITON_CORE_REPO_TAG:STRING=r23.12" "-DTRITON_BACKEND_REPO_TAG:STRING=r23.12" "-DTRITON_ENABLE_GPU:BOOL=OFF" "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" ..
make -j16 VERBOSE=1 install
mkdir -p /data/github_codes/server/build/opt/tritonserver/backends
rm -fr /data/github_codes/server/build/opt/tritonserver/backends/python
cp -r /data/github_codes/server/build/python/install/backends/python /data/github_codes/server/build/opt/tritonserver/backends
#
# end 'python' backend
########

########
# 'onnxruntime' backend
# Delete this section to remove backend from build
#
mkdir -p /data/github_codes/server/build
cd /data/github_codes/server/build
rm -fr onnxruntime
if [[ ! -e onnxruntime ]]; then
  git clone --recursive --single-branch --depth=1 -b r23.12 https://github.com/triton-inference-server/onnxruntime_backend.git onnxruntime;
fi
mkdir -p /data/github_codes/server/build/onnxruntime/build
cd /data/github_codes/server/build/onnxruntime/build
cmake "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DTRITON_BUILD_ONNXRUNTIME_VERSION=1.16.3" "-DTRITON_BUILD_CONTAINER_VERSION=23.12" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/data/github_codes/server/build/onnxruntime/install" "-DTRITON_COMMON_REPO_TAG:STRING=r23.12" "-DTRITON_CORE_REPO_TAG:STRING=r23.12" "-DTRITON_BACKEND_REPO_TAG:STRING=r23.12" "-DTRITON_ENABLE_GPU:BOOL=OFF" "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" ..
make -j16 VERBOSE=1 install
mkdir -p /data/github_codes/server/build/opt/tritonserver/backends
rm -fr /data/github_codes/server/build/opt/tritonserver/backends/onnxruntime
cp -r /data/github_codes/server/build/onnxruntime/install/backends/onnxruntime /data/github_codes/server/build/opt/tritonserver/backends
#
# end 'onnxruntime' backend
########
```

### 3-1 编译triton core相关内容
```
########
# Triton core library and tritonserver executable
#
mkdir -p /data/github_codes/server/build/tritonserver/build
cd /data/github_codes/server/build/tritonserver/build
cmake "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/data/github_codes/server/build/tritonserver/install" "-DTRITON_VERSION:STRING=2.41.0" "-DTRITON_COMMON_REPO_TAG:STRING=r23.12" "-DTRITON_CORE_REPO_TAG:STRING=r23.12" "-DTRITON_BACKEND_REPO_TAG:STRING=r23.12" "-DTRITON_THIRD_PARTY_REPO_TAG:STRING=r23.12" "-DTRITON_ENABLE_LOGGING:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" "-DTRITON_ENABLE_METRICS_GPU:BOOL=OFF" "-DTRITON_ENABLE_METRICS_CPU:BOOL=ON" "-DTRITON_ENABLE_TRACING:BOOL=ON" "-DTRITON_ENABLE_NVTX:BOOL=OFF" "-DTRITON_ENABLE_GPU:BOOL=OFF" "-DTRITON_MIN_COMPUTE_CAPABILITY=6.0" "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_GRPC:BOOL=ON" "-DTRITON_ENABLE_HTTP:BOOL=ON" "-DTRITON_ENABLE_SAGEMAKER:BOOL=OFF" "-DTRITON_ENABLE_VERTEX_AI:BOOL=OFF" "-DTRITON_ENABLE_GCS:BOOL=OFF" "-DTRITON_ENABLE_S3:BOOL=OFF" "-DTRITON_ENABLE_AZURE_STORAGE:BOOL=OFF" "-DTRITON_ENABLE_ENSEMBLE:BOOL=ON" "-DTRITON_ENABLE_TENSORRT:BOOL=OFF" /data/github_codes/server
make -j16 VERBOSE=1 install
mkdir -p /data/github_codes/server/build/opt/tritonserver/bin
cp /data/github_codes/server/build/tritonserver/install/bin/tritonserver /data/github_codes/server/build/opt/tritonserver/bin
mkdir -p /data/github_codes/server/build/opt/tritonserver/lib
cp /data/github_codes/server/build/tritonserver/install/lib/libtritonserver.so /data/github_codes/server/build/opt/tritonserver/lib
mkdir -p /data/github_codes/server/build/opt/tritonserver/python
cp /data/github_codes/server/build/tritonserver/install/python/tritonserver*.whl /data/github_codes/server/build/opt/tritonserver/python
mkdir -p /data/github_codes/server/build/opt/tritonserver/include/triton
cp -r /data/github_codes/server/build/tritonserver/install/include/triton/core /data/github_codes/server/build/opt/tritonserver/include/triton/core
cp /data/github_codes/server/LICENSE /data/github_codes/server/build/opt/tritonserver
cp /data/github_codes/server/TRITON_VERSION /data/github_codes/server/build/opt/tritonserver
#
# end Triton core library and tritonserver executable
########

```

### 3-2 编译python后端代码
```
# 需要修改/data/github_codes/server/build/python 中CMakeLists.txt
# boost下载失败, 将
# https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
# 替换为
# https://jaist.dl.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.gz


########
# 'python' backend
# Delete this section to remove backend from build
#
mkdir -p /data/github_codes/server/build
cd /data/github_codes/server/build
rm -fr python
if [[ ! -e python ]]; then
  git clone --recursive --single-branch --depth=1 -b r23.12 https://github.com/triton-inference-server/python_backend.git python;
fi
mkdir -p /data/github_codes/server/build/python/build
cd /data/github_codes/server/build/python/build
cmake "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/data/github_codes/server/build/python/install" "-DTRITON_COMMON_REPO_TAG:STRING=r23.12" "-DTRITON_CORE_REPO_TAG:STRING=r23.12" "-DTRITON_BACKEND_REPO_TAG:STRING=r23.12" "-DTRITON_ENABLE_GPU:BOOL=OFF" "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" ..
make -j16 VERBOSE=1 install
mkdir -p /data/github_codes/server/build/opt/tritonserver/backends
rm -fr /data/github_codes/server/build/opt/tritonserver/backends/python
cp -r /data/github_codes/server/build/python/install/backends/python /data/github_codes/server/build/opt/tritonserver/backends
#
# end 'python' backend
########
```

### 3-3 编译onnxruntime后端代码
```
# 需要改动/data/github_codes/server/build/onnxruntime中CMakeLists.txt 和 tools/gen_ort_dockerfile.py两个文件
# ///////////////////CMakeLists.txt////////////////////////
onnxruntime后端代码CMakeLists.txt中将gpu开关关闭，dockers镜像地址替换为arm64架构的22.04镜像地址
将
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
替换为
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
将
set(TRITON_BUILD_CONTAINER "nvcr.io/nvidia/tritonserver:${TRITON_BUILD_CONTAINER_VERSION}-py3-min")
替换为
set(TRITON_BUILD_CONTAINER "webhippie/ubuntu:22.04-arm64")

# /////////////////tools/gen_ort_dockerfile.py/////////////
# onnxruntime后端代码tools/gen_ort_dockerfile.py
# 替换软件源，增加下面的代码
# # The Onnx Runtime dockerfile is the collection of steps in
# # https://github.com/microsoft/onnxruntime/tree/master/dockerfiles
# RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list    // 增加

# 增加编译开关，在下面代码处增加
#     if os.name == "posix":
#         if os.getuid() == 0:
#             ep_flags += " --allow_running_as_root"
#     ep_flags += "--allow_running_as_root"   // 增加


########
# 'onnxruntime' backend
# Delete this section to remove backend from build
#
mkdir -p /data/github_codes/server/build
cd /data/github_codes/server/build
rm -fr onnxruntime
if [[ ! -e onnxruntime ]]; then
  git clone --recursive --single-branch --depth=1 -b r23.12 https://github.com/triton-inference-server/onnxruntime_backend.git onnxruntime;
fi
mkdir -p /data/github_codes/server/build/onnxruntime/build
cd /data/github_codes/server/build/onnxruntime/build
cmake "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DTRITON_BUILD_ONNXRUNTIME_VERSION=1.16.3" "-DTRITON_BUILD_CONTAINER_VERSION=23.12" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/data/github_codes/server/build/onnxruntime/install" "-DTRITON_COMMON_REPO_TAG:STRING=r23.12" "-DTRITON_CORE_REPO_TAG:STRING=r23.12" "-DTRITON_BACKEND_REPO_TAG:STRING=r23.12" "-DTRITON_ENABLE_GPU:BOOL=OFF" "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" ..
make -j16 VERBOSE=1 install
mkdir -p /data/github_codes/server/build/opt/tritonserver/backends
rm -fr /data/github_codes/server/build/opt/tritonserver/backends/onnxruntime
cp -r /data/github_codes/server/build/onnxruntime/install/backends/onnxruntime /data/github_codes/server/build/opt/tritonserver/backends
#
# end 'onnxruntime' backend
########
```

### 3-4 编译rknn后端代码
```
########
# 'rknn' backend
# Delete this section to remove backend from build
#
mkdir -p /data/github_codes/server/build
cd /data/github_codes/server/build
rm -fr rknn
if [[ ! -e rknn ]]; then
  git clone -b r23.12 https://github.com/zjd1988/rknn_backend.git rknn;
fi
mkdir -p /data/github_codes/server/build/rknn/build
cd /data/github_codes/server/build/rknn/build
cmake \
    "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" \
    "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DTRITON_BUILD_CONTAINER_VERSION=23.12" \
    "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/data/github_codes/server/build/rknn/install" \
    "-DTRITON_COMMON_REPO_TAG:STRING=r23.12" "-DTRITON_CORE_REPO_TAG:STRING=r23.12" \
    "-DTRITON_BACKEND_REPO_TAG:STRING=r23.12" "-DTRITON_ENABLE_GPU:BOOL=OFF" \
    "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" \
    "-DTRITON_ENABLE_METRICS:BOOL=ON" ..
#make -j16 VERBOSE=1 install
make -j install
mkdir -p /data/github_codes/server/build/opt/tritonserver/backends
rm -fr /data/github_codes/server/build/opt/tritonserver/backends/rknn
cp -r /data/github_codes/server/build/rknn/install/backends/rknn /data/github_codes/server/build/opt/tritonserver/backends
#
# end 'rknn' backend
########
```

### 5 本地测试npu是否可用
```
cd /data/github_codes
git clone -b v1.5.2 https://github.com/rockchip-linux/rknpu2.git
cd ./rknpu2/examples/rknn_yolov5_demo
./build-linux_RK3588.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/github_codes/rknpu2/examples/rknn_yolov5_demo/install/rknn_yolov5_demo_Linux/lib/
./install/rknn_yolov5_demo_Linux/rknn_yolov5_demo model/RK3588/yolov5s-640-640.rknn model/bus.jpg


post process config: box_conf_threshold = 0.25, nms_threshold = 0.45
Read model/bus.jpg ...
img width = 640, img height = 640
Loading mode...
sdk version: 1.5.2 (c6b7b351a@2023-08-23T15:28:22) driver version: 0.8.2
model input num: 1, output num: 3
  index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, w_stride = 640, size_with_stride=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
  index=0, name=output, n_dims=4, dims=[1, 255, 80, 80], n_elems=1632000, size=1632000, w_stride = 0, size_with_stride=1638400, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003860
  index=1, name=283, n_dims=4, dims=[1, 255, 40, 40], n_elems=408000, size=408000, w_stride = 0, size_with_stride=491520, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
  index=2, name=285, n_dims=4, dims=[1, 255, 20, 20], n_elems=102000, size=102000, w_stride = 0, size_with_stride=163840, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003915
model is NHWC input fmt
model input height=640, width=640, channel=3
once run use 22.727000 ms
loadLabelName ./model/coco_80_labels_list.txt
person @ (209 244 286 506) 0.884139
person @ (478 238 559 526) 0.867678
person @ (110 238 230 534) 0.824685
bus @ (94 129 553 468) 0.705055
person @ (79 354 122 516) 0.339254
loop count = 10 , average run  20.872800 ms

```

