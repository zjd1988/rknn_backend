import copy
import numpy as np
import cv2
import heapq
import multiprocessing
from multiprocessing import Process
import tritonclient.http as httpclient
import tritonclient.grpc as grpclient

def multi_input_work_func(pid):
    # generate random inputs
    input1 = np.random.randint(0, 256, size=(1, 128, 128, 1), dtype=np.uint8)
    input2 = np.random.randint(0, 256, size=(1, 128, 128, 3), dtype=np.uint8)
    
    # http client 
    # triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')
  
    # 1 set inputs
    # inputs = []
    # inputs.append(httpclient.InferInput('input.1', input1.shape, "UINT8"))
    # inputs.append(httpclient.InferInput('input', input2.shape, "UINT8"))
    # inputs[0].set_data_from_numpy(input1, binary_data=False)
    # inputs[1].set_data_from_numpy(input2, binary_data=False)

    # 2 set outputs
    # outputs = []
    # outputs.append(httpclient.InferRequestedOutput("8"))

    # grpc client
    triton_client = grpclient.InferenceServerClient(url='127.0.0.1:8001')

    # 1 set inputs
    inputs = []
    inputs.append(grpclient.InferInput('input.1', input1.shape, "UINT8"))
    inputs.append(grpclient.InferInput('input', input2.shape, "UINT8"))
    inputs[0].set_data_from_numpy(input1)
    inputs[1].set_data_from_numpy(input2)

    # 2 set outputs
    outputs = []
    outputs.append(grpclient.InferRequestedOutput("8"))

    # Inference
    print('--> Running model')
    count = 0
    while count < 1:
        results = triton_client.infer(model_name="multi_input", inputs=inputs, outputs=outputs)
        print("{}: infer {}".format(pid, count))
        count = count + 1

    # print result shape
    print(results.as_numpy("8").shape)


if __name__ == '__main__':
    process_num = 1
    process_list = []
    # 创建进程
    for pid in range(process_num):
        p = Process(target=multi_input_work_func, args=(pid,))
        p.start()
        process_list.append(p)
    for j in process_list:
        j.join()
