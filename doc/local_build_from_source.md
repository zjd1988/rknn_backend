### 1 拉取triton-server仓库 
```
# 以我本地的硬件和环境为例(使用orange pi 5b, 镜像版本Orangepi5b_1.0.4_ubuntu_jammy_server_linux5.10.110.7z),
# 交叉编译暂时没尝试

cd /data/github_codes
git clone https://github.com/triton-inference-server/server.git
cd /data/github_codes/server/ && git checkout r23.12
```

### 2 切换到server目录，执行python ./build.py 生成cmake_build编译脚本
```
# 生成不需要docker镜像的编译脚本,--build-dir 使用完整路径(以/data/github_codes/server/build_test为例)
# 注意需要使能enable-mali-gpu，否则无法支持npu多实例

cd /data/github_codes/server
python ./build.py -v --dryrun --no-container-build --backend=ensemble \
--backend=python --backend=onnxruntime --endpoint=grpc --endpoint=http --enable-logging \
--enable-stats --enable-metrics --enable-cpu-metrics --enable-tracing \
--enable-mali-gpu --build-dir=/data/github_codes/server/build
```

### 3 拉取rknn_backend仓库到build路径下
```
cd build
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
    "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DTRITON_BUILD_CONTAINER_VERSION=23.12" \
    "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/data/github_codes/server/build_test/rknn/install" \
    "-DTRITON_COMMON_REPO_TAG:STRING=r23.12" "-DTRITON_CORE_REPO_TAG:STRING=r23.12" \
    "-DTRITON_BACKEND_REPO_TAG:STRING=r23.12" "-DTRITON_ENABLE_GPU:BOOL=OFF" \
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

exit 0

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

########
# 'rknn' backend Delete this section to remove backend from build
#
mkdir -p /tmp/tritonbuild
cd /tmp/tritonbuild
rm -fr rknn
if [[ ! -e rknn ]]; then
  git clone https://github.com/zjd1988/rknn_backend.git rknn;
fi
mkdir -p /tmp/tritonbuild/rknn/build
cd /tmp/tritonbuild/rknn/build
cmake \
    "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" \
    "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DTRITON_BUILD_CONTAINER_VERSION=23.12" \
    "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/tmp/tritonbuild/rknn/install" \
    "-DTRITON_COMMON_REPO_TAG:STRING=r23.12" "-DTRITON_CORE_REPO_TAG:STRING=r23.12" \
    "-DTRITON_BACKEND_REPO_TAG:STRING=r23.12" "-DTRITON_ENABLE_GPU:BOOL=OFF" \
    "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" \
    "-DTRITON_ENABLE_METRICS:BOOL=ON" ..
#make -j1 VERBOSE=1 install
make -j install
mkdir -p /tmp/tritonbuild/install/backends
rm -fr /tmp/tritonbuild/install/backends/rknn
cp -r /tmp/tritonbuild/rknn/install/backends/rknn /tmp/tritonbuild/install/backends
#
# end 'rknn' backend
########

########
# Collect Triton CI artifacts
#
mkdir -p /tmp/tritonbuild/ci
cp -r /workspace/qa /tmp/tritonbuild/ci
cp -r /workspace/deploy /tmp/tritonbuild/ci
mkdir -p /tmp/tritonbuild/ci/docs
cp -r /workspace/docs/examples /tmp/tritonbuild/ci/docs
mkdir -p /tmp/tritonbuild/ci/src/test
cp -r /workspace/src/test/models /tmp/tritonbuild/ci/src/test
cp -r /tmp/tritonbuild/tritonserver/install/bin /tmp/tritonbuild/ci
mkdir -p /tmp/tritonbuild/ci/lib
cp /tmp/tritonbuild/tritonserver/install/lib/libtritonrepoagent_relocation.so /tmp/tritonbuild/ci/lib
cp -r /tmp/tritonbuild/tritonserver/install/python /tmp/tritonbuild/ci
mkdir -p /tmp/tritonbuild/ci/backends
if [[ -e /tmp/tritonbuild/identity/install/backends/identity ]]; then
cp -r /tmp/tritonbuild/identity/install/backends/identity /tmp/tritonbuild/ci/backends
fi
if [[ -e /tmp/tritonbuild/repeat/install/backends/repeat ]]; then
cp -r /tmp/tritonbuild/repeat/install/backends/repeat /tmp/tritonbuild/ci/backends
fi
if [[ -e /tmp/tritonbuild/square/install/backends/square ]]; then
cp -r /tmp/tritonbuild/square/install/backends/square /tmp/tritonbuild/ci/backends
fi
mkdir -p /tmp/tritonbuild/ci/tritonbuild/tritonserver/backends
if [[ -e /tmp/tritonbuild/tritonserver/install/backends/query ]]; then
cp -r /tmp/tritonbuild/tritonserver/install/backends/query /tmp/tritonbuild/ci/tritonbuild/tritonserver/backends
fi
if [[ -e /tmp/tritonbuild/tritonserver/install/backends/implicit_state ]]; then
cp -r /tmp/tritonbuild/tritonserver/install/backends/implicit_state /tmp/tritonbuild/ci/tritonbuild/tritonserver/backends
fi
if [[ -e /tmp/tritonbuild/tritonserver/install/backends/sequence ]]; then
cp -r /tmp/tritonbuild/tritonserver/install/backends/sequence /tmp/tritonbuild/ci/tritonbuild/tritonserver/backends
fi
if [[ -e /tmp/tritonbuild/tritonserver/install/backends/dyna_sequence ]]; then
cp -r /tmp/tritonbuild/tritonserver/install/backends/dyna_sequence /tmp/tritonbuild/ci/tritonbuild/tritonserver/backends
fi
if [[ -e /tmp/tritonbuild/tritonserver/install/backends/distributed_addsub ]]; then
cp -r /tmp/tritonbuild/tritonserver/install/backends/distributed_addsub /tmp/tritonbuild/ci/tritonbuild/tritonserver/backends
fi
if [[ -e /tmp/tritonbuild/tritonserver/install/backends/iterative_sequence ]]; then
cp -r /tmp/tritonbuild/tritonserver/install/backends/iterative_sequence /tmp/tritonbuild/ci/tritonbuild/tritonserver/backends
fi
mkdir -p /tmp/tritonbuild/ci/qa/L0_custom_ops
cp /tmp/tritonbuild/onnxruntime/install/test/libcustom_op_library.so /tmp/tritonbuild/ci/qa/L0_custom_ops
cp /tmp/tritonbuild/onnxruntime/install/test/custom_op_test.onnx /tmp/tritonbuild/ci/qa/L0_custom_ops
cp -r /tmp/tritonbuild/onnxruntime/test/* /tmp/tritonbuild/ci/qa
mkdir -p /tmp/tritonbuild/ci/tritonbuild
rm -fr /tmp/tritonbuild/python/build
rm -fr /tmp/tritonbuild/python/install
cp -r /tmp/tritonbuild/python /tmp/tritonbuild/ci/tritonbuild

rm -fr /tmp/tritonbuild/rknn/build
rm -fr /tmp/tritonbuild/rknn/install
cp -r /tmp/tritonbuild/rknn /tmp/tritonbuild/ci/tritonbuild
#
# end Triton CI artifacts
########

chmod -R a+rw /tmp/tritonbuild/install
chmod -R a+rw /tmp/tritonbuild/ci
```
### 6 切换到server目录下，执行编译
```
cd ../
./build_test/cmake_build

```

### 7 python 后端编译问题
```
需要修改CMakeLists.txt
boost下载失败, 将
https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
替换为
https://jaist.dl.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.gz
```

### 8 onnxruntime 后端编译问题
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

## 9 onnxruntime CMakeLists.txt 完整代码(option)
```
# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(tritononnxruntimebackend LANGUAGES C CXX)

#
# Options
#
# To build the ONNX Runtime backend you must either:
#
#   - Point to an already built ONNX Runtime using
#     TRITON_ONNXRUNTIME_INCLUDE_PATHS and
#     TRITON_ONNXRUNTIME_LIB_PATHS
#
#   or:
#
#   - Set TRITON_BUILD_ONNXRUNTIME_VERSION to the version of ONNX
#     Runtime that you want to be built for the backend.
#
#   - Set TRITON_BUILD_CONTAINER to the Triton container to use as a
#     base for the build. On linux you can instead set
#     TRITON_BUILD_CONTAINER_VERSION to the Triton version that you
#     want to target with the build and the corresponding container
#     from NGC will be used.
#
#   - Optionally set TRITON_BUILD_CUDA_VERSION and
#     TRITON_BUILD_CUDA_HOME. If not set these are automatically set
#     by using the standard cuda install location. For example on
#     windows these will be automatically set based on CUDA_PATH, for
#     example:
#
#         TRITON_BUILD_CUDA_VERSION=11.1
#         TRITON_BUILD_CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\v11.1"
#
#   - If you want TensorRT support set
#     TRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON and set TRITON_BUILD_TENSORRT_HOME.
#
#     Optionally set TRITON_ONNX_TENSORRT_REPO_TAG to specify a branch in https://github.com/onnx/onnx-tensorrt repo
#     example:
#         TRITON_ONNX_TENSORRT_REPO_TAG=master
#     This enables using a version of tensorrt which is not yet supported in ONNXRuntime release branch.
#     By default we pick the default branch which comes with the requested version of onnxruntime.
#
#     Optionally set TRT_VERSION to specify the version of TRT which is being used.
#     This along with TRITON_BUILD_ONNXRUNTIME_VERSION is used to pick the right onnx tensorrt parser version.
#     When TRITON_ONNX_TENSORRT_REPO_TAG is set TRT_VERSION is ignored.
#     When neither TRITON_ONNX_TENSORRT_REPO_TAG or TRT_VERSION are set
#     the default parser version which comes with ORT is picked.
#
#   - If you want OpenVINO support set
#     TRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON and set
#     TRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION to the OpenVino
#     version that is compatible with the specified version of ONNX
#     Runtime.
#
#   - Optionally set TRITON_BUILD_TARGET_PLATFORM to either linux, windows or
#     igpu. If not set, the current platform will be used. If building on
#     Jetpack, always set to igpu to avoid misdetection.
#
#   - If you want to disable GPU usage, set TRITON_ENABLE_GPU=OFF.
#    This will make builds with CUDA and TensorRT flags to fail.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_ONNXRUNTIME_TENSORRT
  "Enable TensorRT execution provider for ONNXRuntime backend in server" OFF)
option(TRITON_ENABLE_ONNXRUNTIME_OPENVINO
  "Enable OpenVINO execution provider for ONNXRuntime backend in server" OFF)
set(TRITON_BUILD_CONTAINER "" CACHE STRING "Triton container to use a base for build")
set(TRITON_BUILD_CONTAINER_VERSION "" CACHE STRING "Triton container version to target")
set(TRITON_BUILD_ONNXRUNTIME_VERSION "" CACHE STRING "ONNXRuntime version to build")
set(TRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION "" CACHE STRING "ONNXRuntime OpenVINO version to build")
set(TRITON_BUILD_TARGET_PLATFORM "" CACHE STRING "Target platform for ONNXRuntime build")
set(TRITON_BUILD_CUDA_VERSION "" CACHE STRING "Version of CUDA install")
set(TRITON_BUILD_CUDA_HOME "" CACHE PATH "Path to CUDA install")
set(TRITON_BUILD_CUDNN_HOME "" CACHE PATH "Path to CUDNN install")
set(TRITON_BUILD_TENSORRT_HOME "" CACHE PATH "Path to TensorRT install")
set(TRITON_ONNXRUNTIME_INCLUDE_PATHS "" CACHE PATH "Paths to ONNXRuntime includes")
set(TRITON_ONNX_TENSORRT_REPO_TAG "" CACHE STRING "Tag for onnx-tensorrt repo")
set(TRT_VERSION "" CACHE STRING "TRT version for this build.")
set(TRITON_ONNXRUNTIME_LIB_PATHS "" CACHE PATH "Paths to ONNXRuntime libraries")

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

if (WIN32)
  if(TRITON_ENABLE_ONNXRUNTIME_OPENVINO)
    message(FATAL_ERROR
      "TRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON not supported for Windows")
  endif()
endif() # WIN32

if (NOT TRITON_ENABLE_GPU)
  if (TRITON_ENABLE_ONNXRUNTIME_TENSORRT)
    message(FATAL_ERROR "TRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON requires TRITON_ENABLE_GPU=ON")
  endif() # TRITON_ENABLE_ONNXRUNTIME_TENSORRT
endif() # NOT TRITON_ENABLE_GPU

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TRITON_ONNXRUNTIME_DOCKER_BUILD OFF)
if(TRITON_ONNXRUNTIME_LIB_PATHS STREQUAL "")
  set(TRITON_ONNXRUNTIME_DOCKER_BUILD ON)
endif()

message(STATUS "Using Onnxruntime docker: ${TRITON_ONNXRUNTIME_DOCKER_BUILD}")

if(NOT TRITON_ONNXRUNTIME_DOCKER_BUILD)
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS ${TRITON_ONNXRUNTIME_LIB_PATHS})
  if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
    find_library(OV_LIBRARY
      NAMES openvino
      PATHS ${TRITON_ONNXRUNTIME_LIB_PATHS})
  endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

else()

  if(NOT TRITON_BUILD_CONTAINER AND NOT TRITON_BUILD_CONTAINER_VERSION)
    message(FATAL_ERROR
      "TRITON_BUILD_ONNXRUNTIME_VERSION requires TRITON_BUILD_CONTAINER or TRITON_BUILD_CONTAINER_VERSION")
  endif()

  if(NOT TRITON_BUILD_CONTAINER)
    #set(TRITON_BUILD_CONTAINER "nvcr.io/nvidia/tritonserver:${TRITON_BUILD_CONTAINER_VERSION}-py3-min")
    set(TRITON_BUILD_CONTAINER "webhippie/ubuntu:22.04-arm64")
  endif()

  set(TRITON_ONNXRUNTIME_DOCKER_IMAGE "tritonserver_onnxruntime")
  set(TRITON_ONNXRUNTIME_DOCKER_MEMORY "8g")
  set(TRITON_ONNXRUNTIME_INCLUDE_PATHS "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/include")
  set(TRITON_ONNXRUNTIME_LIB_PATHS "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib")
  if (WIN32)
    set(ONNXRUNTIME_LIBRARY "onnxruntime")
  else()
    set(ONNXRUNTIME_LIBRARY "libonnxruntime.so")
  endif() # WIN32
  if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
    set(OV_LIBRARY "libopenvino.so")
  endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO
endif()

#
# Dependencies
#
# FetchContent's composability isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
endif() # TRITON_ENABLE_GPU

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_onnxruntime.ldscript libtriton_onnxruntime.ldscript COPYONLY)

add_library(
  triton-onnxruntime-backend SHARED
  src/onnxruntime.cc
  src/onnxruntime_loader.cc
  src/onnxruntime_loader.h
  src/onnxruntime_utils.cc
  src/onnxruntime_utils.h
)

add_library(
  TritonOnnxRuntimeBackend::triton-onnxruntime-backend ALIAS triton-onnxruntime-backend
)

target_include_directories(
  triton-onnxruntime-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TRITON_ONNXRUNTIME_INCLUDE_PATHS}
)

target_compile_features(triton-onnxruntime-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-onnxruntime-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor>
)

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    triton-onnxruntime-backend
    PRIVATE TRITON_ENABLE_GPU=1
  )
endif() # TRITON_ENABLE_GPU
if(${TRITON_ENABLE_ONNXRUNTIME_TENSORRT})
  target_compile_definitions(
    triton-onnxruntime-backend
    PRIVATE TRITON_ENABLE_ONNXRUNTIME_TENSORRT=1
  )
endif() # TRITON_ENABLE_ONNXRUNTIME_TENSORRT
if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
  target_compile_definitions(
    triton-onnxruntime-backend
    PRIVATE TRITON_ENABLE_ONNXRUNTIME_OPENVINO=1
  )
endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

if (WIN32)
set_target_properties(
  triton-onnxruntime-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_onnxruntime
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
)
else ()
set_target_properties(
  triton-onnxruntime-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_onnxruntime
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_onnxruntime.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_onnxruntime.ldscript"
)
endif()

FOREACH(p ${TRITON_ONNXRUNTIME_LIB_PATHS})
  target_link_directories(
    triton-onnxruntime-backend
    PRIVATE ${p}
  )
ENDFOREACH(p)

target_link_libraries(
  triton-onnxruntime-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    ${TRITON_ONNXRUNTIME_LDFLAGS}
    ${ONNXRUNTIME_LIBRARY}
)

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-onnxruntime-backend
    PRIVATE
      CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
  target_link_libraries(
    triton-onnxruntime-backend
    PRIVATE
      ${OV_LIBRARY}
  )
endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

#
# Build the ONNX Runtime libraries using docker.
#
if(TRITON_ONNXRUNTIME_DOCKER_BUILD)
  set(_GEN_FLAGS "")
  if(NOT ${TRITON_BUILD_TARGET_PLATFORM} STREQUAL "")
    set(_GEN_FLAGS ${_GEN_FLAGS} "--target-platform=${TRITON_BUILD_TARGET_PLATFORM}")
  endif() # TRITON_BUILD_TARGET_PLATFORM
  if(NOT ${TRITON_BUILD_CUDA_VERSION} STREQUAL "")
    set(_GEN_FLAGS ${_GEN_FLAGS} "--cuda-version=${TRITON_BUILD_CUDA_VERSION}")
  endif() # TRITON_BUILD_CUDA_VERSION
  if(NOT ${TRITON_BUILD_CUDA_HOME} STREQUAL "")
    set(_GEN_FLAGS ${_GEN_FLAGS} "--cuda-home=${TRITON_BUILD_CUDA_HOME}")
  endif() # TRITON_BUILD_CUDA_HOME
  if(NOT ${TRITON_BUILD_CUDNN_HOME} STREQUAL "")
    set(_GEN_FLAGS ${_GEN_FLAGS} "--cudnn-home=${TRITON_BUILD_CUDNN_HOME}")
  endif() # TRITON_BUILD_CUDNN_HOME
  if(NOT ${TRITON_BUILD_TENSORRT_HOME} STREQUAL "")
    set(_GEN_FLAGS ${_GEN_FLAGS} "--tensorrt-home=${TRITON_BUILD_TENSORRT_HOME}")
  endif() # TRITON_BUILD_TENSORRT_HOME
  if(${TRITON_ENABLE_ONNXRUNTIME_TENSORRT})
    set(_GEN_FLAGS ${_GEN_FLAGS} "--ort-tensorrt")
  endif() # TRITON_ENABLE_ONNXRUNTIME_TENSORRT
  if(${TRITON_ENABLE_ONNXRUNTIME_OPENVINO})
    set(_GEN_FLAGS ${_GEN_FLAGS} "--ort-openvino=${TRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION}")
  endif() # TRITON_ENABLE_ONNXRUNTIME_OPENVINO

  set(ENABLE_GPU_EXTRA_ARGS "")
  if(${TRITON_ENABLE_GPU})
    set(ENABLE_GPU_EXTRA_ARGS "--enable-gpu")
  endif() # TRITON_ENABLE_GPU

  if (WIN32)
    add_custom_command(
      OUTPUT
        onnxruntime/lib/${ONNXRUNTIME_LIBRARY}
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_ort_dockerfile.py --triton-container="${TRITON_BUILD_CONTAINER}" --ort-version="${TRITON_BUILD_ONNXRUNTIME_VERSION}" --trt-version="${TRT_VERSION}" --onnx-tensorrt-tag="${TRITON_ONNX_TENSORRT_REPO_TAG}" ${_GEN_FLAGS} --output=Dockerfile.ort ${ENABLE_GPU_EXTRA_ARGS}
      COMMAND docker build --memory ${TRITON_ONNXRUNTIME_DOCKER_MEMORY} --cache-from=${TRITON_ONNXRUNTIME_DOCKER_IMAGE} --cache-from=${TRITON_ONNXRUNTIME_DOCKER_IMAGE}_cache0 --cache-from=${TRITON_ONNXRUNTIME_DOCKER_IMAGE}_cache1 -t ${TRITON_ONNXRUNTIME_DOCKER_IMAGE} -f ./Dockerfile.ort ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND powershell.exe -noprofile -c "docker rm onnxruntime_backend_ort > $null 2>&1; if ($LASTEXITCODE) { 'error ignored...' }; exit 0"
      COMMAND docker create --name onnxruntime_backend_ort ${TRITON_ONNXRUNTIME_DOCKER_IMAGE}
      COMMAND rmdir /s/q onnxruntime
      COMMAND docker cp onnxruntime_backend_ort:/opt/onnxruntime onnxruntime
      COMMAND docker rm onnxruntime_backend_ort
      COMMENT "Building ONNX Runtime"
    )
  else()
    add_custom_command(
      OUTPUT
        onnxruntime/lib/${ONNXRUNTIME_LIBRARY}
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_ort_dockerfile.py  --ort-build-config="${CMAKE_BUILD_TYPE}" --triton-container="${TRITON_BUILD_CONTAINER}" --ort-version="${TRITON_BUILD_ONNXRUNTIME_VERSION}" --trt-version="${TRT_VERSION}" --onnx-tensorrt-tag="${TRITON_ONNX_TENSORRT_REPO_TAG}" ${_GEN_FLAGS} --output=Dockerfile.ort ${ENABLE_GPU_EXTRA_ARGS}
      COMMAND docker build --cache-from=${TRITON_ONNXRUNTIME_DOCKER_IMAGE} --cache-from=${TRITON_ONNXRUNTIME_DOCKER_IMAGE}_cache0 --cache-from=${TRITON_ONNXRUNTIME_DOCKER_IMAGE}_cache1 -t ${TRITON_ONNXRUNTIME_DOCKER_IMAGE} -f ./Dockerfile.ort ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND docker rm onnxruntime_backend_ort || echo 'error ignored...' || true
      COMMAND docker create --name onnxruntime_backend_ort ${TRITON_ONNXRUNTIME_DOCKER_IMAGE}
      COMMAND rm -fr onnxruntime
      COMMAND docker cp onnxruntime_backend_ort:/opt/onnxruntime onnxruntime
      COMMAND docker rm onnxruntime_backend_ort
      COMMENT "Building ONNX Runtime"
    )
  endif() # WIN32

  add_custom_target(ort_target DEPENDS onnxruntime/lib/${ONNXRUNTIME_LIBRARY})
  add_library(onnxruntime-library SHARED IMPORTED GLOBAL)
  add_dependencies(onnxruntime-library ort_target)
  add_dependencies(triton-onnxruntime-backend onnxruntime-library)

  if (WIN32)
    set_target_properties(
      onnxruntime-library
      PROPERTIES
        IMPORTED_LOCATION onnxruntime/bin/${ONNXRUNTIME_LIBRARY}
    )
  else()
    set_target_properties(
      onnxruntime-library
      PROPERTIES
        IMPORTED_LOCATION onnxruntime/lib/${ONNXRUNTIME_LIBRARY}
    )
  endif() # WIN32
endif() # TRITON_ONNXRUNTIME_DOCKER_BUILD

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonOnnxRuntimeBackend)

install(
  TARGETS
    triton-onnxruntime-backend
  EXPORT
    triton-onnxruntime-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
)

# For Jetson, we build the onnxruntime backend once and re-use
# that tar file. We copy over the libraries and other requirements
# prior to running this build and therefore these set of install
# commands are not needed.
if(TRITON_ONNXRUNTIME_DOCKER_BUILD)
  install(
    DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
    PATTERN *lib EXCLUDE
    PATTERN *bin EXCLUDE
    PATTERN *include EXCLUDE
    PATTERN *test EXCLUDE
  )

  install(
    DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/bin/
    USE_SOURCE_PERMISSIONS
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
  )

  if (NOT WIN32)
    install(
      DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/lib/
      USE_SOURCE_PERMISSIONS
      DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/onnxruntime
    )

    install(
      DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/test
      USE_SOURCE_PERMISSIONS
      DESTINATION ${CMAKE_INSTALL_PREFIX}
    )
  endif() # NOT WIN32
endif() # TRITON_ONNXRUNTIME_DOCKER_BUILD

install(
  EXPORT
    triton-onnxruntime-backend-targets
  FILE
    TritonOnnxRuntimeBackendTargets.cmake
  NAMESPACE
    TritonOnnxRuntimeBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonOnnxRuntimeBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOnnxRuntimeBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOnnxRuntimeBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-onnxruntime-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonOnnxRuntimeBackendTargets.cmake
  NAMESPACE TritonOnnxRuntimeBackend::
)

export(PACKAGE TritonOnnxRuntimeBackend)
```

## 10 onnxruntim tools/gen_ort_dockerfile.py 完整代码(option)
```
完整代码如下
#!/usr/bin/env python3
# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import platform
import re

FLAGS = None

ORT_TO_TRTPARSER_VERSION_MAP = {
    "1.9.0": (
        "8.2",  # TensorRT version
        "release/8.2-GA",  # ONNX-Tensorrt parser version
    ),
    "1.10.0": (
        "8.2",  # TensorRT version
        "release/8.2-GA",  # ONNX-Tensorrt parser version
    ),
}


def target_platform():
    if FLAGS.target_platform is not None:
        return FLAGS.target_platform
    return platform.system().lower()


def dockerfile_common():
    df = """
ARG BASE_IMAGE={}
ARG ONNXRUNTIME_VERSION={}
ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ARG ONNXRUNTIME_BUILD_CONFIG={}
""".format(
        FLAGS.triton_container, FLAGS.ort_version, FLAGS.ort_build_config
    )

    if FLAGS.ort_openvino is not None:
        df += """
ARG ONNXRUNTIME_OPENVINO_VERSION={}
""".format(
            FLAGS.ort_openvino
        )

    df += """
FROM ${BASE_IMAGE}
WORKDIR /workspace
"""
    return df


def dockerfile_for_linux(output_file):
    df = dockerfile_common()
    df += """
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

# The Onnx Runtime dockerfile is the collection of steps in
# https://github.com/microsoft/onnxruntime/tree/master/dockerfiles
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        zip \
        ca-certificates \
        build-essential \
        curl \
        libcurl4-openssl-dev \
        libssl-dev \
        patchelf \
        python3-dev \
        python3-pip \
        git \
        gnupg \
        gnupg1

# Install dependencies from
# onnxruntime/dockerfiles/scripts/install_common_deps.sh.
RUN apt update -q=2 \\
    && apt install -y gpg wget \\
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \\
    && . /etc/os-release \\
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \\
    && apt-get update -q=2 \\
    && apt-get install -y --no-install-recommends cmake=3.27.7* cmake-data=3.27.7* \\
    && cmake --version

"""
    if FLAGS.enable_gpu:
        df += """
# Allow configure to pick up cuDNN where it expects it.
# (Note: $CUDNN_VERSION is defined by base image)
RUN _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2) && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/include && \
    ln -s /usr/include/cudnn.h /usr/local/cudnn-$_CUDNN_VERSION/cuda/include/cudnn.h && \
    mkdir -p /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64 && \
    ln -s /etc/alternatives/libcudnn_so /usr/local/cudnn-$_CUDNN_VERSION/cuda/lib64/libcudnn.so
"""

    if FLAGS.ort_openvino is not None:
        df += """
# Install OpenVINO
ARG ONNXRUNTIME_OPENVINO_VERSION
ENV INTEL_OPENVINO_DIR /opt/intel/openvino_${ONNXRUNTIME_OPENVINO_VERSION}

# Step 1: Download and install core components
# Ref: https://docs.openvino.ai/2023.0/openvino_docs_install_guides_installing_openvino_from_archive_linux.html#step-1-download-and-install-the-openvino-core-components
RUN curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_ubuntu22_2023.0.0.10926.b4452d56304_x86_64.tgz --output openvino_${ONNXRUNTIME_OPENVINO_VERSION}.tgz && \
    tar -xf openvino_${ONNXRUNTIME_OPENVINO_VERSION}.tgz && \
    mkdir -p ${INTEL_OPENVINO_DIR} && \
    mv l_openvino_toolkit_ubuntu22_2023.0.0.10926.b4452d56304_x86_64/* ${INTEL_OPENVINO_DIR} && \
    rm openvino_${ONNXRUNTIME_OPENVINO_VERSION}.tgz && \
    (cd ${INTEL_OPENVINO_DIR}/install_dependencies && \
        ./install_openvino_dependencies.sh -y) && \
    ln -s ${INTEL_OPENVINO_DIR} ${INTEL_OPENVINO_DIR}/../openvino_`echo ${ONNXRUNTIME_OPENVINO_VERSION} | awk '{print substr($0,0,4)}'`

# Step 2: Configure the environment
# Ref: https://docs.openvino.ai/2023.0/openvino_docs_install_guides_installing_openvino_from_archive_linux.html#step-2-configure-the-environment
ENV InferenceEngine_DIR=$INTEL_OPENVINO_DIR/runtime/cmake
ENV ngraph_DIR=$INTEL_OPENVINO_DIR/runtime/cmake
ENV OpenVINO_DIR=$INTEL_OPENVINO_DIR/runtime/cmake
ENV LD_LIBRARY_PATH $INTEL_OPENVINO_DIR/runtime/lib/intel64:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=$INTEL_OPENVINO_DIR/runtime/lib/intel64/pkgconfig
ENV PYTHONPATH $INTEL_OPENVINO_DIR/python/python3.10:$INTEL_OPENVINO_DIR/python/python3:$PYTHONPATH
"""

    ## TEMPORARY: Using the tensorrt-8.0 branch until ORT 1.9 release to enable ORT backend with TRT 8.0 support.
    # For ORT versions 1.8.0 and below the behavior will remain same. For ORT version 1.8.1 we will
    # use tensorrt-8.0 branch instead of using rel-1.8.1
    # From ORT 1.9 onwards we will switch back to using rel-* branches
    if FLAGS.ort_version == "1.8.1":
        df += """
    #
    # ONNX Runtime build
    #
    ARG ONNXRUNTIME_VERSION
    ARG ONNXRUNTIME_REPO
    ARG ONNXRUNTIME_BUILD_CONFIG

    RUN git clone -b tensorrt-8.0 --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
        (cd onnxruntime && git submodule update --init --recursive)

       """
    # Use the tensorrt-8.5ea branch to use Tensor RT 8.5a to use the built-in tensorrt parser
    elif FLAGS.ort_version == "1.12.1":
        df += """
    #
    # ONNX Runtime build
    #
    ARG ONNXRUNTIME_VERSION
    ARG ONNXRUNTIME_REPO
    ARG ONNXRUNTIME_BUILD_CONFIG

    RUN git clone -b tensorrt-8.5ea --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
        (cd onnxruntime && git submodule update --init --recursive)

       """
    else:
        df += """
    #
    # ONNX Runtime build
    #
    ARG ONNXRUNTIME_VERSION
    ARG ONNXRUNTIME_REPO
    ARG ONNXRUNTIME_BUILD_CONFIG

    RUN git clone -b rel-${ONNXRUNTIME_VERSION} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
        (cd onnxruntime && git submodule update --init --recursive)

        """

    if FLAGS.onnx_tensorrt_tag != "":
        df += """
    RUN (cd /workspace/onnxruntime/cmake/external/onnx-tensorrt && git fetch origin {}:ortrefbranch && git checkout ortrefbranch)
    """.format(
            FLAGS.onnx_tensorrt_tag
        )

    ep_flags = ""
    if FLAGS.enable_gpu:
        ep_flags = "--use_cuda"
        if FLAGS.cuda_version is not None:
            ep_flags += ' --cuda_version "{}"'.format(FLAGS.cuda_version)
        if FLAGS.cuda_home is not None:
            ep_flags += ' --cuda_home "{}"'.format(FLAGS.cuda_home)
        if FLAGS.cudnn_home is not None:
            ep_flags += ' --cudnn_home "{}"'.format(FLAGS.cudnn_home)
        elif target_platform() == "igpu":
            ep_flags += ' --cudnn_home "/usr/lib/aarch64-linux-gnu"'
        if FLAGS.ort_tensorrt:
            ep_flags += " --use_tensorrt"
            if FLAGS.ort_version >= "1.12.1":
                ep_flags += " --use_tensorrt_builtin_parser"
            if FLAGS.tensorrt_home is not None:
                ep_flags += ' --tensorrt_home "{}"'.format(FLAGS.tensorrt_home)

    if os.name == "posix":
        if os.getuid() == 0:
            ep_flags += " --allow_running_as_root"

    if FLAGS.ort_openvino is not None:
        ep_flags += " --use_openvino CPU_FP32"

    if target_platform() == "igpu":
        ep_flags += (
            " --skip_tests --cmake_extra_defines 'onnxruntime_BUILD_UNIT_TESTS=OFF'"
        )
        cuda_archs = "53;62;72;87"
    else:
        cuda_archs = "60;61;70;75;80;86;90"

    df += """
WORKDIR /workspace/onnxruntime
ARG COMMON_BUILD_ARGS="--config ${{ONNXRUNTIME_BUILD_CONFIG}} --skip_submodule_sync --parallel --build_shared_lib \
    --build_dir /workspace/build --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES='{}' "
""".format(
        cuda_archs
    )

    df += """
RUN ./build.sh ${{COMMON_BUILD_ARGS}} --update --build {}
""".format(
        ep_flags
    )

    df += """
#
# Copy all artifacts needed by the backend to /opt/onnxruntime
#
WORKDIR /opt/onnxruntime

RUN mkdir -p /opt/onnxruntime && \
    cp /workspace/onnxruntime/LICENSE /opt/onnxruntime && \
    cat /workspace/onnxruntime/cmake/external/onnx/VERSION_NUMBER > /opt/onnxruntime/ort_onnx_version.txt

# ONNX Runtime headers, libraries and binaries
RUN mkdir -p /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h \
       /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h \
       /opt/onnxruntime/include && \
    cp /workspace/onnxruntime/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h \
       /opt/onnxruntime/include

RUN mkdir -p /opt/onnxruntime/lib && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_shared.so \
       /opt/onnxruntime/lib && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime.so \
       /opt/onnxruntime/lib
"""
    if target_platform() == "igpu":
        df += """
RUN mkdir -p /opt/onnxruntime/bin
"""
    else:
        df += """
RUN mkdir -p /opt/onnxruntime/bin && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/onnxruntime_perf_test \
       /opt/onnxruntime/bin && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/onnx_test_runner \
       /opt/onnxruntime/bin && \
    (cd /opt/onnxruntime/bin && chmod a+x *)
"""

    if FLAGS.enable_gpu:
        df += """
RUN cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_cuda.so \
       /opt/onnxruntime/lib
"""

    if FLAGS.ort_tensorrt:
        df += """
# TensorRT specific headers and libraries
RUN cp /workspace/onnxruntime/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h \
       /opt/onnxruntime/include && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_tensorrt.so \
       /opt/onnxruntime/lib
"""

    if FLAGS.ort_openvino is not None:
        df += """
# OpenVino specific headers and libraries
RUN cp -r ${INTEL_OPENVINO_DIR}/docs/licensing /opt/onnxruntime/LICENSE.openvino

RUN cp /workspace/onnxruntime/include/onnxruntime/core/providers/openvino/openvino_provider_factory.h \
       /opt/onnxruntime/include

RUN cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libonnxruntime_providers_openvino.so \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino.so.${ONNXRUNTIME_OPENVINO_VERSION} \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino_c.so.${ONNXRUNTIME_OPENVINO_VERSION} \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino_intel_cpu_plugin.so \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino_ir_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} \
       /opt/onnxruntime/lib && \
    cp ${INTEL_OPENVINO_DIR}/runtime/lib/intel64/libopenvino_onnx_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} \
       /opt/onnxruntime/lib && \
    cp /usr/lib/x86_64-linux-gnu/libtbb.so.12 /opt/onnxruntime/lib && \
    cp /usr/lib/x86_64-linux-gnu/libpugixml.so.1 /opt/onnxruntime/lib

RUN OV_SHORT_VERSION=`echo ${ONNXRUNTIME_OPENVINO_VERSION} | awk '{ split($0,a,"."); print substr(a[1],3) a[2] a[3] }'` && \
    (cd /opt/onnxruntime/lib && \
        chmod a-x * && \
        ln -s libopenvino.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino.so.${OV_SHORT_VERSION} && \
        ln -s libopenvino.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino.so && \
        ln -s libopenvino_c.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_c.so.${OV_SHORT_VERSION} && \
        ln -s libopenvino_c.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_c.so && \
        ln -s libopenvino_ir_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_ir_frontend.so.${OV_SHORT_VERSION} && \
        ln -s libopenvino_ir_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_ir_frontend.so && \
        ln -s libopenvino_onnx_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_onnx_frontend.so.${OV_SHORT_VERSION} && \
        ln -s libopenvino_onnx_frontend.so.${ONNXRUNTIME_OPENVINO_VERSION} libopenvino_onnx_frontend.so)
"""
    # Linking compiled ONNX Runtime libraries to their corresponding versioned libraries
    df += """
RUN cd /opt/onnxruntime/lib \
        && ln -s libonnxruntime.so libonnxruntime.so.${ONNXRUNTIME_VERSION}
"""
    df += """
RUN cd /opt/onnxruntime/lib && \
    for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
    done

# For testing copy ONNX custom op library and model
"""
    if target_platform() == "igpu":
        df += """
RUN mkdir -p /opt/onnxruntime/test
"""
    else:
        df += """
RUN mkdir -p /opt/onnxruntime/test && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/libcustom_op_library.so \
       /opt/onnxruntime/test && \
    cp /workspace/build/${ONNXRUNTIME_BUILD_CONFIG}/testdata/custom_op_library/custom_op_test.onnx \
       /opt/onnxruntime/test
"""

    with open(output_file, "w") as dfile:
        dfile.write(df)


def dockerfile_for_windows(output_file):
    df = dockerfile_common()

    ## TEMPORARY: Using the tensorrt-8.0 branch until ORT 1.9 release to enable ORT backend with TRT 8.0 support.
    # For ORT versions 1.8.0 and below the behavior will remain same. For ORT version 1.8.1 we will
    # use tensorrt-8.0 branch instead of using rel-1.8.1
    # From ORT 1.9 onwards we will switch back to using rel-* branches
    if FLAGS.ort_version == "1.8.1":
        df += """
SHELL ["cmd", "/S", "/C"]

#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO

RUN git clone -b tensorrt-8.0 --recursive %ONNXRUNTIME_REPO% onnxruntime && \
    (cd onnxruntime && git submodule update --init --recursive)
"""
    else:
        df += """
SHELL ["cmd", "/S", "/C"]

#
# ONNX Runtime build
#
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_REPO
RUN git clone -b rel-%ONNXRUNTIME_VERSION% --recursive %ONNXRUNTIME_REPO% onnxruntime && \
    (cd onnxruntime && git submodule update --init --recursive)
"""

    if FLAGS.onnx_tensorrt_tag != "":
        df += """
    RUN (cd \\workspace\\onnxruntime\\cmake\\external\\onnx-tensorrt && git fetch origin {}:ortrefbranch && git checkout ortrefbranch)
    """.format(
            FLAGS.onnx_tensorrt_tag
        )

    ep_flags = ""
    if FLAGS.enable_gpu:
        ep_flags = "--use_cuda"
        if FLAGS.cuda_version is not None:
            ep_flags += ' --cuda_version "{}"'.format(FLAGS.cuda_version)
        if FLAGS.cuda_home is not None:
            ep_flags += ' --cuda_home "{}"'.format(FLAGS.cuda_home)
        if FLAGS.cudnn_home is not None:
            ep_flags += ' --cudnn_home "{}"'.format(FLAGS.cudnn_home)
        if FLAGS.ort_tensorrt:
            ep_flags += " --use_tensorrt"
            if FLAGS.tensorrt_home is not None:
                ep_flags += ' --tensorrt_home "{}"'.format(FLAGS.tensorrt_home)
    if FLAGS.ort_openvino is not None:
        ep_flags += " --use_openvino CPU_FP32"

    df += """
WORKDIR /workspace/onnxruntime
ARG VS_DEVCMD_BAT="\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
RUN powershell Set-Content 'build.bat' -value 'call %VS_DEVCMD_BAT%',(Get-Content 'build.bat')
RUN build.bat --cmake_generator "Visual Studio 17 2022" --config Release --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=60;61;70;75;80;86;90" --skip_submodule_sync --parallel --build_shared_lib --update --build --build_dir /workspace/build {}
""".format(
        ep_flags
    )

    df += """
#
# Copy all artifacts needed by the backend to /opt/onnxruntime
#
WORKDIR /opt/onnxruntime
RUN copy \\workspace\\onnxruntime\\LICENSE \\opt\\onnxruntime
RUN copy \\workspace\\onnxruntime\\cmake\\external\\onnx\\VERSION_NUMBER \\opt\\onnxruntime\\ort_onnx_version.txt

# ONNX Runtime headers, libraries and binaries
WORKDIR /opt/onnxruntime/include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\session\\onnxruntime_c_api.h \\opt\\onnxruntime\\include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\session\\onnxruntime_session_options_config_keys.h \\opt\\onnxruntime\\include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\providers\\cpu\\cpu_provider_factory.h \\opt\\onnxruntime\\include

WORKDIR /opt/onnxruntime/bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime.dll \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_shared.dll \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_perf_test.exe \\opt\\onnxruntime\\bin
RUN copy \\workspace\\build\\Release\\Release\\onnx_test_runner.exe \\opt\\onnxruntime\\bin

WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime.lib \\opt\\onnxruntime\\lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_shared.lib \\opt\\onnxruntime\\lib
"""

    if FLAGS.enable_gpu:
        df += """
WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_cuda.lib \\opt\\onnxruntime\\lib
WORKDIR /opt/onnxruntime/bin
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_cuda.dll \\opt\\onnxruntime\\bin
"""

    if FLAGS.ort_tensorrt:
        df += """
# TensorRT specific headers and libraries
WORKDIR /opt/onnxruntime/include
RUN copy \\workspace\\onnxruntime\\include\\onnxruntime\\core\\providers\\tensorrt\\tensorrt_provider_factory.h \\opt\\onnxruntime\\include

WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_tensorrt.dll \\opt\\onnxruntime\\bin

WORKDIR /opt/onnxruntime/lib
RUN copy \\workspace\\build\\Release\\Release\\onnxruntime_providers_tensorrt.lib \\opt\\onnxruntime\\lib
"""
    with open(output_file, "w") as dfile:
        dfile.write(df)


def preprocess_gpu_flags():
    if target_platform() == "windows":
        # Default to CUDA based on CUDA_PATH envvar and TensorRT in
        # C:/tensorrt
        if "CUDA_PATH" in os.environ:
            if FLAGS.cuda_home is None:
                FLAGS.cuda_home = os.environ["CUDA_PATH"]
            elif FLAGS.cuda_home != os.environ["CUDA_PATH"]:
                print("warning: --cuda-home does not match CUDA_PATH envvar")

        if FLAGS.cudnn_home is None:
            FLAGS.cudnn_home = FLAGS.cuda_home

        version = None
        m = re.match(r".*v([1-9]?[0-9]+\.[0-9]+)$", FLAGS.cuda_home)
        if m:
            version = m.group(1)

        if FLAGS.cuda_version is None:
            FLAGS.cuda_version = version
        elif FLAGS.cuda_version != version:
            print("warning: --cuda-version does not match CUDA_PATH envvar")

        if (FLAGS.cuda_home is None) or (FLAGS.cuda_version is None):
            print("error: windows build requires --cuda-version and --cuda-home")

        if FLAGS.tensorrt_home is None:
            FLAGS.tensorrt_home = "/tensorrt"
    else:
        if "CUDNN_VERSION" in os.environ:
            version = None
            m = re.match(r"([0-9]\.[0-9])\.[0-9]\.[0-9]", os.environ["CUDNN_VERSION"])
            if m:
                version = m.group(1)
            if FLAGS.cudnn_home is None:
                FLAGS.cudnn_home = "/usr/local/cudnn-{}/cuda".format(version)

        if FLAGS.cuda_home is None:
            FLAGS.cuda_home = "/usr/local/cuda"

        if (FLAGS.cuda_home is None) or (FLAGS.cudnn_home is None):
            print("error: linux build requires --cudnn-home and --cuda-home")

        if FLAGS.tensorrt_home is None:
            FLAGS.tensorrt_home = "/usr/src/tensorrt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--triton-container",
        type=str,
        required=True,
        help="Triton base container to use for ORT build.",
    )
    parser.add_argument("--ort-version", type=str, required=True, help="ORT version.")
    parser.add_argument(
        "--output", type=str, required=True, help="File to write Dockerfile to."
    )
    parser.add_argument(
        "--enable-gpu", action="store_true", required=False, help="Enable GPU support"
    )
    parser.add_argument(
        "--ort-build-config",
        type=str,
        default="Release",
        choices=["Debug", "Release", "RelWithDebInfo"],
        help="ORT build configuration.",
    )
    parser.add_argument(
        "--target-platform",
        required=False,
        default=None,
        help='Target for build, can be "linux", "windows" or "igpu". If not specified, build targets the current platform.',
    )

    parser.add_argument(
        "--cuda-version", type=str, required=False, help="Version for CUDA."
    )
    parser.add_argument(
        "--cuda-home", type=str, required=False, help="Home directory for CUDA."
    )
    parser.add_argument(
        "--cudnn-home", type=str, required=False, help="Home directory for CUDNN."
    )
    parser.add_argument(
        "--ort-openvino",
        type=str,
        required=False,
        help="Enable OpenVino execution provider using specified OpenVINO version.",
    )
    parser.add_argument(
        "--ort-tensorrt",
        action="store_true",
        required=False,
        help="Enable TensorRT execution provider.",
    )
    parser.add_argument(
        "--tensorrt-home", type=str, required=False, help="Home directory for TensorRT."
    )
    parser.add_argument(
        "--onnx-tensorrt-tag", type=str, default="", help="onnx-tensorrt repo tag."
    )
    parser.add_argument("--trt-version", type=str, default="", help="TRT version.")

    FLAGS = parser.parse_args()
    if FLAGS.enable_gpu:
        preprocess_gpu_flags()

    # if a tag is provided by the user, then simply use it
    # if the tag is empty - check whether there is an entry in the ORT_TO_TRTPARSER_VERSION_MAP
    # map corresponding to ort version + trt version combo. If yes then use it
    # otherwise we leave it empty and use the defaults from ort
    if (
        FLAGS.onnx_tensorrt_tag == ""
        and FLAGS.ort_version in ORT_TO_TRTPARSER_VERSION_MAP.keys()
    ):
        trt_version = re.match(r"^[0-9]+\.[0-9]+", FLAGS.trt_version)
        if (
            trt_version
            and trt_version.group(0)
            == ORT_TO_TRTPARSER_VERSION_MAP[FLAGS.ort_version][0]
        ):
            FLAGS.onnx_tensorrt_tag = ORT_TO_TRTPARSER_VERSION_MAP[FLAGS.ort_version][1]

    if target_platform() == "windows":
        # OpenVINO EP not yet supported for windows build
        if FLAGS.ort_openvino is not None:
            print("warning: OpenVINO not supported for windows, ignoring")
            FLAGS.ort_openvino = None
        dockerfile_for_windows(FLAGS.output)
    else:
        dockerfile_for_linux(FLAGS.output)
```

