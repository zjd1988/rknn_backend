# rknn_backend
## 编译步骤
### 1 拉取triton-server仓库 
```
# 以我本地的目录和环境为例

cd /data/github_codes
git clone https://github.com/triton-inference-server/server.git
```

### 2 切换到server目录，执行python ./build.py 生成cmake_build编译脚本
```
#生成不需要docker镜像的编译脚本,--build-dir 使用完整路径(以/data/github_codes/server/build_test为例)
#注意需要使能enable-mali-gpu，否则无法支持npu多实例

cd /data/github_codes/server
python ./build.py -v --dryrun --no-container-build --backend=ensemble \
--endpoint=grpc --endpoint=http --enable-logging \
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
#make -j1 VERBOSE=1 install
make -j install
mkdir -p /data/github_codes/server/build_test/opt/tritonserver/backends
rm -fr /data/github_codes/server/build_test/opt/tritonserver/backends/rknn
cp -r /data/github_codes/server/build_test/rknn/install/backends/rknn /data/github_codes/server/build_test/opt/tritonserver/backends
#
# end 'rknn' backend
########
```
### 4 切换到server目录下，执行编译
```
cd ../
./build_test/cmake_build

# 最终会在build_test/opt/tritonserver目录下存放编译生成的所有内容
```


