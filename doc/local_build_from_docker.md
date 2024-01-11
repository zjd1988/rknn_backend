## 1 拉取triton-server仓库 
```
# 以我本地的硬件和环境为例(使用orange pi 5b, 镜像版本Orangepi5b_1.0.4_ubuntu_jammy_server_linux5.10.110.7z),
# 该镜像已经预装了docker等软件，可以直接使用

cd /data/github_codes
git clone https://github.com/triton-inference-server/server.git
cd server && git checkout r23.12
```

## 2 切换到server目录，执行python ./build.py 生成docker_build等编译脚本
```
# 生成需要docker的编译脚本
# 注意需要使能enable-mali-gpu，否则无法支持npu多实例

cd /data/github_codes/server
python ./build.py -v --dryrun --backend=ensemble \
--backend=python --backend=onnxruntime --endpoint=grpc --endpoint=http --enable-logging \
--enable-stats --enable-metrics --enable-cpu-metrics --enable-tracing \
--enable-mali-gpu

```
## 3 查看docker_build脚本cat /data/github_codes/server/build/docker_build
```
#!/usr/bin/env bash

#
# Docker-based build script for Triton Inference Server
#

# Exit script immediately if any command fails
set -e
set -x

# step1 构建tritonserver_buildbase 镜像
########
# Create Triton base build image
# This image contains all dependencies necessary to build Triton
#
cd /data/github_codes/server
docker build -t tritonserver_buildbase -f /data/github_codes/server/build/Dockerfile.buildbase --pull --cache-from=tritonserver_buildbase --cache-from=tritonserver_buildbase_cache0 --cache-from=tritonserver_buildbase_cache1 .

# step2 基于tritonserver_buildbase 镜像, 调用cmake_build脚本, 编译各种后端代码
########
# Run build in tritonserver_buildbase container
# Mount a directory into the container where the install
# artifacts will be placed.
#
if [ "$(docker ps -a | grep tritonserver_builder)" ]; then  docker rm -f tritonserver_builder; fi
docker run -w /workspace/build --name tritonserver_builder -it -v /var/run/docker.sock:/var/run/docker.sock tritonserver_buildbase ./cmake_build
docker cp tritonserver_builder:/tmp/tritonbuild/install /data/github_codes/server/build
docker cp tritonserver_builder:/tmp/tritonbuild/ci /data/github_codes/server/build

# step3 基于编译各种后端代码生成库, 生成最终的镜像文件
########
# Create final tritonserver image
#
cd /data/github_codes/server
docker build -t tritonserver -f /data/github_codes/server/build/Dockerfile .

# step4 可以忽略
########
# Create CI base image
#
cd /data/github_codes/server
docker build -t tritonserver_cibase -f /data/github_codes/server/build/Dockerfile.cibase .
```

## 4 执行docker_build脚本
```
cd /data/github_codes/server
因为运行过程中会报错，所以会依次执行./build/docker_build文件的命令
```
### 4-1 step1 构建tritonserver_buildbase 镜像
```
cd /data/github_codes/server
docker build -t tritonserver_buildbase -f /data/github_codes/server/build/Dockerfile.buildbase --pull --cache-from=tritonserver_buildbase --cache-from=tritonserver_buildbase_cache0 --cache-from=tritonserver_buildbase_cache1 .
```

### 4-2 step2 基于tritonserver_buildbase 镜像, 调用cmake_build脚本, 编译各种后端代码
```
# 因为cmake_build 执行过程依然会出现问题，所以通过在tritonserver_buildbase容器内分步骤执行cmake_build编译后端代码
```

#### 4-2-1 进入tritonserver_buildbase镜像
```
docker run -w /workspace/build --name tritonserver_builder -it -v /var/run/docker.sock:/var/run/docker.sock tritonserver_buildbase /bin/bash
```

#### 4-2-2 编译triton_core代码
```
########
# Triton core library and tritonserver executable
#
mkdir -p /tmp/tritonbuild/tritonserver/build
cd /tmp/tritonbuild/tritonserver/build
cmake "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/tmp/tritonbuild/tritonserver/install" "-DTRITON_VERSION:STRING=2.41.0" "-DTRITON_COMMON_REPO_TAG:STRING=r23.12" "-DTRITON_CORE_REPO_TAG:STRING=r23.12" "-DTRITON_BACKEND_REPO_TAG:STRING=r23.12" "-DTRITON_THIRD_PARTY_REPO_TAG:STRING=r23.12" "-DTRITON_ENABLE_LOGGING:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" "-DTRITON_ENABLE_METRICS_GPU:BOOL=OFF" "-DTRITON_ENABLE_METRICS_CPU:BOOL=ON" "-DTRITON_ENABLE_TRACING:BOOL=ON" "-DTRITON_ENABLE_NVTX:BOOL=OFF" "-DTRITON_ENABLE_GPU:BOOL=OFF" "-DTRITON_MIN_COMPUTE_CAPABILITY=6.0" "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_GRPC:BOOL=ON" "-DTRITON_ENABLE_HTTP:BOOL=ON" "-DTRITON_ENABLE_SAGEMAKER:BOOL=OFF" "-DTRITON_ENABLE_VERTEX_AI:BOOL=OFF" "-DTRITON_ENABLE_GCS:BOOL=OFF" "-DTRITON_ENABLE_S3:BOOL=OFF" "-DTRITON_ENABLE_AZURE_STORAGE:BOOL=OFF" "-DTRITON_ENABLE_ENSEMBLE:BOOL=ON" "-DTRITON_ENABLE_TENSORRT:BOOL=OFF" /workspace
make -j16 VERBOSE=1 install
mkdir -p /tmp/tritonbuild/install/bin
cp /tmp/tritonbuild/tritonserver/install/bin/tritonserver /tmp/tritonbuild/install/bin
mkdir -p /tmp/tritonbuild/install/lib
cp /tmp/tritonbuild/tritonserver/install/lib/libtritonserver.so /tmp/tritonbuild/install/lib
mkdir -p /tmp/tritonbuild/install/python
cp /tmp/tritonbuild/tritonserver/install/python/tritonserver*.whl /tmp/tritonbuild/install/python
mkdir -p /tmp/tritonbuild/install/include/triton
cp -r /tmp/tritonbuild/tritonserver/install/include/triton/core /tmp/tritonbuild/install/include/triton/core
cp /workspace/LICENSE /tmp/tritonbuild/install
cp /workspace/TRITON_VERSION /tmp/tritonbuild/install
mkdir -p /tmp/tritonbuild/install/third-party-src
cd /tmp/tritonbuild/tritonserver/build
tar zcf /tmp/tritonbuild/install/third-party-src/src.tar.gz third-party-src
cp /workspace/docker/README.third-party-src /tmp/tritonbuild/install/third-party-src/README
#
# end Triton core library and tritonserver executable
########
```

#### 4-2-3 编译python后端代码
```
# 编译过程中, 会报boost下载失败. 需要修改/tmp/tritonbuild/python 的CMakeLists.txt文件
# 将 https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
# 替换为
# https://jaist.dl.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.gz

########
# 'python' backend
# Delete this section to remove backend from build
#
mkdir -p /tmp/tritonbuild
cd /tmp/tritonbuild
rm -fr python
if [[ ! -e python ]]; then
  git clone --recursive --single-branch --depth=1 -b r23.12 https://github.com/triton-inference-server/python_backend.git python;
fi
mkdir -p /tmp/tritonbuild/python/build
cd /tmp/tritonbuild/python/build
cmake "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/tmp/tritonbuild/python/install" "-DTRITON_COMMON_REPO_TAG:STRING=r23.12" "-DTRITON_CORE_REPO_TAG:STRING=r23.12" "-DTRITON_BACKEND_REPO_TAG:STRING=r23.12" "-DTRITON_ENABLE_GPU:BOOL=OFF" "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" ..
make -j16 VERBOSE=1 install
mkdir -p /tmp/tritonbuild/install/backends
rm -fr /tmp/tritonbuild/install/backends/python
cp -r /tmp/tritonbuild/python/install/backends/python /tmp/tritonbuild/install/backends
#
# end 'python' backend
########
```

#### 4-2-4 编译onnxruntime后端代码
```
# 需要改动两个文件/tmp/tritonbuild/onnxruntime/CMakeLists.txt 和 /tmp/tritonbuild/onnxruntime/tools/gen_ort_dockerfile.py两个文件
# /////////////////////////////////////////////////////////
# ///////////////////CMakeLists.txt////////////////////////
# /////////////////////////////////////////////////////////
# onnxruntime后端代码CMakeLists.txt中将gpu开关关闭，dockers镜像地址替换为arm64架构的22.04镜像地址
# 将
# option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
# 替换为
# option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
# 将
# set(TRITON_BUILD_CONTAINER "nvcr.io/nvidia/tritonserver:${TRITON_BUILD_CONTAINER_VERSION}-py3-min")
# 替换为
# set(TRITON_BUILD_CONTAINER "webhippie/ubuntu:22.04-arm64")

# /////////////////////////////////////////////////////////
# /////////////////tools/gen_ort_dockerfile.py/////////////
# /////////////////////////////////////////////////////////
# onnxruntime后端代码tools/gen_ort_dockerfile.py
# 替换软件源，增加下面的代码
# The Onnx Runtime dockerfile is the collection of steps in
# https://github.com/microsoft/onnxruntime/tree/master/dockerfiles
# RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list    // 增加


########
# 'onnxruntime' backend
# Delete this section to remove backend from build
#
mkdir -p /tmp/tritonbuild
cd /tmp/tritonbuild
rm -fr onnxruntime
if [[ ! -e onnxruntime ]]; then
  git clone --recursive --single-branch --depth=1 -b r23.12 https://github.com/triton-inference-server/onnxruntime_backend.git onnxruntime;
fi
mkdir -p /tmp/tritonbuild/onnxruntime/build
cd /tmp/tritonbuild/onnxruntime/build
cmake "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DTRITON_BUILD_ONNXRUNTIME_VERSION=1.16.3" "-DTRITON_BUILD_CONTAINER_VERSION=23.12" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/tmp/tritonbuild/onnxruntime/install" "-DTRITON_COMMON_REPO_TAG:STRING=r23.12" "-DTRITON_CORE_REPO_TAG:STRING=r23.12" "-DTRITON_BACKEND_REPO_TAG:STRING=r23.12" "-DTRITON_ENABLE_GPU:BOOL=OFF" "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" ..
make -j16 VERBOSE=1 install
mkdir -p /tmp/tritonbuild/install/backends
rm -fr /tmp/tritonbuild/install/backends/onnxruntime
cp -r /tmp/tritonbuild/onnxruntime/install/backends/onnxruntime /tmp/tritonbuild/install/backends
#
# end 'onnxruntime' backend
########
```

#### 4-2-5 编译rknn后端代码
```
########
# 'rknn' backend
# Delete this section to remove backend from build
#
mkdir -p /tmp/tritonbuild
cd /tmp/tritonbuild
rm -fr rknn
if [[ ! -e rknn ]]; then
  git clone -b r23.12 https://github.com/zjd1988/rknn_backend.git rknn;
fi
mkdir -p /tmp/tritonbuild/rknn/build
cd /tmp/tritonbuild/rknn/build
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
mkdir -p /tmp/tritonbuild/install/backends
rm -fr /tmp/tritonbuild/install/backends/rknn
cp -r /tmp/tritonbuild/rknn/install/backends/rknn /tmp/tritonbuild/install/backends
#
# end 'rknn' backend
########
```

### 5 启动triton-server镜像测试npu是否可用
```
cd /data/github_codes
git clone -b v1.5.2 https://github.com/rockchip-linux/rknpu2.git
cd ./rknpu2/examples/rknn_yolov5_demo
./build-linux_RK3588.sh
docker run --privileged -v /dev/:/dev -v /data/github_codes/rknpu2:/workspace --network host -it tritonserver:latest

进入docker内, 执行如下命令
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/examples/rknn_yolov5_demo/install/rknn_yolov5_demo_Linux/lib/
./install/rknn_yolov5_demo_Linux/rknn_yolov5_demo model/RK3588/yolov5s-640-640.rknn model/bus.jpg

# 输出以下内容：
post process config: box_conf_threshold = 0.25, nms_threshold = 0.45
Read model/bus.jpg ...
img width = 640, img height = 640
Loading mode...
sdk version: 1.4.0 (a10f100eb@2022-09-09T09:07:14) driver version: 0.8.2
model input num: 1, output num: 3
  index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
  index=0, name=output, n_dims=5, dims=[1, 3, 85, 80], n_elems=1632000, size=1632000, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=77, scale=0.080445
  index=1, name=371, n_dims=5, dims=[1, 3, 85, 40], n_elems=408000, size=408000, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=56, scale=0.080794
  index=2, name=390, n_dims=5, dims=[1, 3, 85, 20], n_elems=102000, size=102000, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=69, scale=0.081305
model is NHWC input fmt
model input height=640, width=640, channel=3
once run use 56.123000 ms
loadLabelName ./model/coco_80_labels_list.txt
person @ (114 235 212 527) 0.819099
person @ (210 242 284 509) 0.814970
person @ (479 235 561 520) 0.790311
bus @ (99 141 557 445) 0.693320
person @ (78 338 122 520) 0.404960
loop count = 10 , average run  38.620200 ms

```
