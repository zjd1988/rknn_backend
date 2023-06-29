import copy
import numpy as np
import cv2
import heapq
import multiprocessing
from multiprocessing import Process
import tritonclient.http as httpclient
import tritonclient.grpc as grpclient

LABEL_PATH = './labels.txt'
IMG_PATH = './dog_224x224.jpg'
IMG_SIZE = 224

def mobilenet_work_func(pid):
    # read labels
    labels = []
    with open(LABEL_PATH, "r") as f:
        labels = f.readlines()    
    # read image
    img = cv2.imread(IMG_PATH)
    img_1 = copy.deepcopy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape((1, IMG_SIZE, IMG_SIZE, 3))

    # http client 
    # triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')
    # inputs = []
    # inputs.append(httpclient.InferInput('input', img.shape, "UINT8"))
    # inputs[0].set_data_from_numpy(img, binary_data=False)

    # outputs = []
    # outputs.append(httpclient.InferRequestedOutput("MobilenetV1/Predictions/Reshape_1"))

    # set inputs
    triton_client = grpclient.InferenceServerClient(url='127.0.0.1:8001')
    inputs = []
    inputs.append(grpclient.InferInput('input', img.shape, "UINT8"))
    inputs[0].set_data_from_numpy(img)
    # set outputs
    outputs = []
    outputs.append(grpclient.InferRequestedOutput("MobilenetV1/Predictions/Reshape_1"))

    # Inference
    print('--> Running model')
    count = 0
    while count < 1:
        results = triton_client.infer(model_name="single_input", inputs=inputs, outputs=outputs)
        print("{}: infer {}".format(pid, count))
        count = count + 1

    # print infer result
    class_result = results.as_numpy("MobilenetV1/Predictions/Reshape_1").reshape(1001)
    top_index = heapq.nlargest(3, range(len(class_result)), class_result.__getitem__)
    for index in top_index:
        print("{}:{}".format(labels[index].strip("\n"), class_result[index]))


if __name__ == '__main__':    
    process_num = 1
    process_list = []
    for pid in range(process_num):
        p = Process(target=mobilenet_work_func, args=(pid,))
        p.start()
        process_list.append(p)
    for j in process_list:
        j.join()
