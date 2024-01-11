### 1 拉取triton-server仓库和安装相关依赖
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

### 2 切换到server目录，执行python ./build.py 生成cmake_build编译脚本
```
# 生成不需要docker镜像的编译脚本,--build-dir 使用完整路径(以/data/github_codes/server/build为例)
# 注意需要使能enable-mali-gpu，否则无法支持npu多实例

cd /data/github_codes/server
python ./build.py -v --dryrun --no-container-build --backend=ensemble \
--backend=python --backend=onnxruntime --endpoint=grpc --endpoint=http --enable-logging \
--enable-stats --enable-metrics --enable-cpu-metrics --enable-tracing \
--enable-mali-gpu --build-dir=$PWD/build

cat /data/github_codes/server/build/cmake_build
```

### 3 按照cmake_build， 依次编译不同模块

#### 3-1 编译triton core相关内容
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

#### 3-2 编译python后端代码
```
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

#### 3-3 编译onnxruntime后端代码
```
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

#### 3-4 编译rknn后端代码
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

### 4 python 后端编译问题
```
需要修改CMakeLists.txt
boost下载失败, 将
https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
替换为
https://jaist.dl.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.gz
```

### 5 onnxruntime 后端编译问题
```
需要改动两个文件CMakeLists.txt 和 tools/gen_ort_dockerfile.py
/////////////////////////////////////////////////////////
///////////////////CMakeLists.txt////////////////////////
/////////////////////////////////////////////////////////
onnxruntime后端代码CMakeLists.txt中将gpu开关关闭，dockers镜像地址替换为arm64架构的22.04镜像地址
将
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
替换为
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
将
set(TRITON_BUILD_CONTAINER "nvcr.io/nvidia/tritonserver:${TRITON_BUILD_CONTAINER_VERSION}-py3-min")
替换为
set(TRITON_BUILD_CONTAINER "webhippie/ubuntu:22.04-arm64")

/////////////////////////////////////////////////////////
/////////////////tools/gen_ort_dockerfile.py/////////////
/////////////////////////////////////////////////////////
onnxruntime后端代码tools/gen_ort_dockerfile.py
替换为下面，主要是替换了源
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list
```

