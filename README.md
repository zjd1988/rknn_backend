# rknn_backend
## 一 编译步骤
 
### 1 本地源码编译 
参考 [本地源码编译](https://github.com/zjd1988/rknn_backend/tree/main/doc/local_build_from_source.md) 文件

### 2 本地docker编译
参考 [本地Docker编译](https://github.com/zjd1988/rknn_backend/tree/main/doc/local_build_from_docker.md) 文件


## 二 测试
注：适用于源码编译方式，docker编译方式需先挂载代码目录进镜像后再操作
### 1 启动triton-server服务
```
./build/opt/tritonserver/bin/tritonserver  --model-repository /data/github_codes/server/build/rknn/examples/models --backend-directory /data/github_codes/server/build/opt/tritonserver/backends/
```

### 2 yolov5模型测试
```
# 需要预先在python环境安装 tritonclient
python -m pip install tritonclient[all]
cd /data/github_codes/server/build/rknn/examples/yolov5
python test_yolov5.py

# 可以通过修改/data/github_codes/server/build/rknn/examples/models/yolov5/config.pbtxt
# 配置模型实例个数和模型加载的npu核id
# 注：目前，暂时不支持模型core_mask按RKNN_NPU_CORE_0_1和RKNN_NPU_CORE_0_1_2进行加载
```
![yolov5测试结果](https://github.com/zjd1988/rknn_backend/blob/main/examples/yolov5/yolov5_result.jpg)

### 3 single_input (mobilenet) 模型测试
```
cd /data/github_codes/server/build/rknn/examples/single_input
python test_single_input.py

# mobilenet test top3 result
# Shih-Tzu:0.984375
# Pekinese:0.0078125
# Lhasa:0.00390625
```

### 4 multi_input 模型测试
```
cd /data/github_codes/server/build/rknn/examples/multi_input
python test_multi_input.py

# multi input result shape
# (1, 8, 128, 128)
```

### 5 ensemble_mobilenet 模型测试(ensemble+python)
```
cd /data/github_codes/server/build/rknn/examples/ensemble_mobilenet
python test_ensemble_mobilenet.py --image ../single_input/dog_224x224.jpg --label_file ../single_input/labels.txt

# Result is class: Shih-Tzu
```
