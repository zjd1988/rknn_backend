### 1 拉取triton-server仓库 
```
# 以我本地的硬件和环境为例(使用orange pi 5b, 镜像版本Orangepi5b_1.0.4_ubuntu_jammy_server_linux5.10.110.7z),
# 该镜像已经预装了docker等软件，可以直接使用

cd /data/github_codes
git clone https://github.com/triton-inference-server/server.git
```

### 2 切换到server目录，执行python ./build.py 生成docker_build等编译脚本
```
# 生成需要docker的编译脚本
# 注意需要使能enable-mali-gpu，否则无法支持npu多实例

cd /data/github_codes/server
python ./build.py -v --dryrun --backend=ensemble \
--backend=python --endpoint=grpc --endpoint=http --enable-logging \
--enable-stats --enable-metrics --enable-cpu-metrics --enable-tracing \
--enable-mali-gpu 
```

### 3 修改cmake_build，添加rknn_backend相关脚本命令
```
# 执行完成第一步会在/data/github_codes/server/build_test目录下生成cmake_build
# 在该文件最后添加下面的rknn_backend相关编译脚本命令

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
    "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DTRITON_BUILD_CONTAINER_VERSION=23.05" \
    "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/tmp/tritonbuild/rknn/install" \
    "-DTRITON_COMMON_REPO_TAG:STRING=r23.05" "-DTRITON_CORE_REPO_TAG:STRING=r23.05" \
    "-DTRITON_BACKEND_REPO_TAG:STRING=r23.05" "-DTRITON_ENABLE_GPU:BOOL=OFF" \
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
```

### 4 完整的cmake_build编译脚本如下(option)
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
cmake "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" \
    "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DCMAKE_BUILD_TYPE=Release" \
    "-DCMAKE_INSTALL_PREFIX:PATH=/tmp/tritonbuild/tritonserver/install" \
    "-DTRITON_VERSION:STRING=2.34.0" "-DTRITON_COMMON_REPO_TAG:STRING=r23.05" \
    "-DTRITON_CORE_REPO_TAG:STRING=r23.05" "-DTRITON_BACKEND_REPO_TAG:STRING=r23.05" \
    "-DTRITON_THIRD_PARTY_REPO_TAG:STRING=r23.05" "-DTRITON_ENABLE_LOGGING:BOOL=ON" \
    "-DTRITON_ENABLE_STATS:BOOL=ON" "-DTRITON_ENABLE_METRICS:BOOL=ON" "-DTRITON_ENABLE_METRICS_GPU:BOOL=OFF" \
    "-DTRITON_ENABLE_METRICS_CPU:BOOL=ON" "-DTRITON_ENABLE_TRACING:BOOL=ON" "-DTRITON_ENABLE_NVTX:BOOL=OFF" \
    "-DTRITON_ENABLE_GPU:BOOL=OFF" "-DTRITON_MIN_COMPUTE_CAPABILITY=6.0" "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" \
    "-DTRITON_ENABLE_GRPC:BOOL=ON" "-DTRITON_ENABLE_HTTP:BOOL=ON" "-DTRITON_ENABLE_SAGEMAKER:BOOL=OFF" \
    "-DTRITON_ENABLE_VERTEX_AI:BOOL=OFF" "-DTRITON_ENABLE_GCS:BOOL=OFF" "-DTRITON_ENABLE_S3:BOOL=OFF" \
    "-DTRITON_ENABLE_AZURE_STORAGE:BOOL=OFF" "-DTRITON_ENABLE_ENSEMBLE:BOOL=ON" \
    "-DTRITON_ENABLE_TENSORRT:BOOL=OFF" /workspace
make -j16 VERBOSE=1 install
mkdir -p /tmp/tritonbuild/install/bin
cp /tmp/tritonbuild/tritonserver/install/bin/tritonserver /tmp/tritonbuild/install/bin
mkdir -p /tmp/tritonbuild/install/lib
cp /tmp/tritonbuild/tritonserver/install/lib/libtritonserver.so /tmp/tritonbuild/install/lib
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

########
# 'python' backend
# Delete this section to remove backend from build
#
mkdir -p /tmp/tritonbuild
cd /tmp/tritonbuild
rm -fr python
if [[ ! -e python ]]; then
  git clone --recursive --single-branch --depth=1 -b r23.05 https://github.com/triton-inference-server/python_backend.git python;
fi
mkdir -p /tmp/tritonbuild/python/build
cd /tmp/tritonbuild/python/build
cmake \
    "-DTRT_VERSION=${TRT_VERSION}" "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}" \
    "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DCMAKE_BUILD_TYPE=Release" \
    "-DCMAKE_INSTALL_PREFIX:PATH=/tmp/tritonbuild/python/install" \
    "-DTRITON_COMMON_REPO_TAG:STRING=r23.05" "-DTRITON_CORE_REPO_TAG:STRING=r23.05" \
    "-DTRITON_BACKEND_REPO_TAG:STRING=r23.05" "-DTRITON_ENABLE_GPU:BOOL=OFF" \
    "-DTRITON_ENABLE_MALI_GPU:BOOL=ON" "-DTRITON_ENABLE_STATS:BOOL=ON" \
    "-DTRITON_ENABLE_METRICS:BOOL=ON" ..
make -j16 VERBOSE=1 install
mkdir -p /tmp/tritonbuild/install/backends
rm -fr /tmp/tritonbuild/install/backends/python
cp -r /tmp/tritonbuild/python/install/backends/python /tmp/tritonbuild/install/backends
#
# end 'python' backend
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
    "-DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}" "-DTRITON_BUILD_CONTAINER_VERSION=23.05" \
    "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX:PATH=/tmp/tritonbuild/rknn/install" \
    "-DTRITON_COMMON_REPO_TAG:STRING=r23.05" "-DTRITON_CORE_REPO_TAG:STRING=r23.05" \
    "-DTRITON_BACKEND_REPO_TAG:STRING=r23.05" "-DTRITON_ENABLE_GPU:BOOL=OFF" \
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

### 5 切换到server目录下，执行编译
```
cd ../
./build_test/dcoker_build

# 编译耗时较长(1h+), 最终会在生成如下镜像
tritonserver                  latest          f0f4e5a93b32   20 hours ago   584MB
```

### 6 启动triton-server镜像测试
```
cd /data/github_codes
git clone https://github.com/rockchip-linux/rknpu2.git
docker run --privileged -v /dev/:/dev -v /data/github_codes/rknpu2:/workspace --network host -it tritonserver:latest

进入docker内, 执行如下命令
cd /workspace/examples/rknn_yolov5_demo
./build-linux_RK3588.sh
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
once run use 49.000000 ms
loadLabelName ./model/coco_80_labels_list.txt
person @ (114 235 212 527) 0.819099
person @ (210 242 284 509) 0.814970
person @ (479 235 561 520) 0.790311
bus @ (99 141 557 445) 0.693320
person @ (78 338 122 520) 0.404960
loop count = 10 , average run  41.760800 ms

```
