# rknn_backend
## 零 环境依赖
### 1 卸载原来系统安装的python
```
# 如果系统自带python大于3.6没有问题，否则最好进行卸载, 然后用miniconda构建虚拟环境
# 卸载步骤参考：
sudo apt-get remove python3

sudo rm -r /etc/py*
sudo rm -r /usr/bin/py*
sudo rm -r /usr/lib/py*
sudo rm -r /usr/local/lib/py*
sudo rm -r /usr/share/py*
sudo rm -r /usr/share/man/man1/py*
```
### 2 安装miniconda
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

### 3 创建虚拟环境并安装python包
```
conda create -n triton_rknn python=3.10
python -m pip install tritonclient[all]
python -m pip install numpy opencv-python-headless
```

## 一 编译步骤
 
### 1 本地源码编译
在编译前需要激活conda虚拟环境
conda activate triton_rknn
参考 [本地源码编译](https://github.com/zjd1988/rknn_backend/tree/main/doc/local_build_from_source.md) 文件

### 2 本地docker编译
参考 [本地Docker编译](https://github.com/zjd1988/rknn_backend/tree/main/doc/local_build_from_docker.md) 文件


## 二 测试
注：适用于源码编译方式，docker编译方式需先挂载代码目录进镜像后再操作
### 1 启动triton-server服务
```
conda activate triton_rknn
./build/opt/tritonserver/bin/tritonserver  --model-repository /data/github_codes/server/build/rknn/examples/models --backend-directory /data/github_codes/server/build/opt/tritonserver/backends/
```

### 2 yolov5模型测试
```
# 需要预先在python环境安装 tritonclient
python -m pip install tritonclient[all]
cd /data/github_codes/server/build/rknn/examples/yolov5
python test_yolov5.py

--> Running model
0: infer 0
class: person, score: 0.819098949432373
box coordinate left,top,right,down: [114.69233334064484, 235.684387922287, 212.63444888591766, 527.1685173511505]
class: person, score: 0.8149696588516235
box coordinate left,top,right,down: [210.97113871574402, 242.16578316688538, 284.33705830574036, 509.1424443721771]
class: person, score: 0.7903112769126892
box coordinate left,top,right,down: [479.5874242782593, 235.37401449680328, 561.1043481826782, 520.7360318899155]
class: person, score: 0.4049600064754486
box coordinate left,top,right,down: [78.8878903388977, 338.7200300693512, 122.68799161911011, 520.08789229393]
class: bus , score: 0.6933198571205139
box coordinate left,top,right,down: [99.32104778289795, 141.9212429523468, 557.3707246780396, 445.96871066093445]

# 可以通过修改/data/github_codes/server/build/rknn/examples/models/yolov5/config.pbtxt
# 配置模型实例个数和模型加载的npu核id
# 注：目前，暂时不支持模型core_mask按RKNN_NPU_CORE_0_1和RKNN_NPU_CORE_0_1_2进行加载
```
![yolov5测试结果](https://github.com/zjd1988/rknn_backend/blob/main/examples/yolov5/yolov5_result.jpg)

### 3 single_input (mobilenet) 模型测试
```
cd /data/github_codes/server/build/rknn/examples/single_input
python test_single_input.py

--> Running model
0: infer 0
Shih-Tzu:0.984375
Pekinese:0.0078125
Lhasa:0.00390625
```

### 4 multi_input 模型测试
```
cd /data/github_codes/server/build/rknn/examples/multi_input
python test_multi_input.py

--> Running model
0: infer 0
(1, 8, 128, 128)
```

### 5 ensemble_mobilenet 模型测试(ensemble+python)
```
cd /data/github_codes/server/build/rknn/examples/ensemble_mobilenet
python test_ensemble_mobilenet.py --image ../single_input/dog_224x224.jpg --label_file ../single_input/labels.txt

# Result is class: Shih-Tzu
```
