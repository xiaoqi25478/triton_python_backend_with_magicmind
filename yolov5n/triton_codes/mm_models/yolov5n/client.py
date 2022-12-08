from tritonclient.utils import *
import tritonclient.http as httpclient

import argparse
import numpy as np
import cv2
import torch
import sys
import os
import time
import math

def coco_dataset(
    file_list_txt="coco_file_list_5000.txt",
    image_dir="../../../../datasets/coco/val2017",
    count=-1
):
    with open(file_list_txt, "r") as f:
        lines = f.readlines()
    current_count = 0
    for line in lines:
        image_name = line.strip()
        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path)
        yield img, image_path
        current_count += 1
        if current_count >= count and count != -1:
            break

def letterbox(img, dst_shape):
    src_h, src_w = img.shape[0], img.shape[1]
    dst_h, dst_w = dst_shape
    ratio = min(dst_h / src_h, dst_w / src_w)
    unpad_h, unpad_w = int(math.floor(src_h * ratio)), int(math.floor(src_w * ratio))
    if ratio != 1:
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (unpad_w, unpad_h), interp)
    # padding
    pad_t = int(math.floor((dst_h - unpad_h) / 2))
    pad_b = dst_h - unpad_h - pad_t
    pad_l = int(math.floor((dst_w - unpad_w) / 2))
    pad_r = dst_w - unpad_w - pad_l
    img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img, ratio

class Record:
    def __init__(self, filename):
        self.file = open(filename, "w")

    def write(self, line, _print = False):
        self.file.write(line + "\n")
        if _print:
            print(line)
            
parser = argparse.ArgumentParser()
parser.add_argument('--input_width', dest = 'input_width', default = 640, type = int, help = 'model input width')
parser.add_argument('--input_height', dest = 'input_height', default = 640, type = int, help = 'model input height')
parser.add_argument('--batch', dest = 'batch', default = 1, type = int, help = 'model input batch')
parser.add_argument("--save_img", dest="save_img", type=bool, default=False)

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = "/data/datasets/COCO2017/val2017"
    image_num = 5000
    file_list = os.path.join(cur_dir,'../../data/coco_file_list_5000.txt')
    label_path = os.path.join(cur_dir,'../../data/coco.names')
    output_dir = os.path.join(cur_dir,'../../output')
    
    model_name = "yolov5n"
    args = parser.parse_args()
    dataset = coco_dataset(file_list_txt = file_list, image_dir = image_dir, count = image_num)
    img_size = [args.input_width, args.input_height]
    
    print("Client start run ...")
    
    with httpclient.InferenceServerClient("localhost:8000") as client:
        start_time = time.time()
        all_results = []
        for img, img_path in dataset:
            img_name = os.path.splitext(img_path.split("/")[-1])[0]
            print("Inference img : ", img_name)
            # 准备输入数据
            show_img = img
            img, ratio = letterbox(img, img_size)
            # BGR to RGB
            img = img[:, :, ::-1]
            img = np.expand_dims(img, 0) # (1, 640, 640, 3)
            
            input0_data = img.astype(np.uint8)
            inputs = [
                httpclient.InferInput("INPUT0", input0_data.shape,
                                    np_to_triton_dtype(input0_data.dtype))]
            inputs[0].set_data_from_numpy(input0_data)
            outputs = [
            httpclient.InferRequestedOutput("OUTPUT0"),
            httpclient.InferRequestedOutput("OUTPUT1"),
            ]
            response = client.infer(model_name,
                                    inputs,
                                    request_id=str(1),
                                    outputs=outputs)
            result = response.get_response()
            output0_data = response.as_numpy("OUTPUT0")
            output1_data = response.as_numpy("OUTPUT1")
            outputs_np = [output0_data,output1_data]
            
            # 处理输出数据
            pred = torch.from_numpy(output0_data)
            detection_num = torch.from_numpy(output1_data)
            reshape_value = torch.reshape(pred, (-1, 1))
            src_h, src_w = show_img.shape[0], show_img.shape[1]
            scale_w = ratio * src_w 
            scale_h = ratio * src_h

            record = Record(output_dir + "/" + img_name + '.txt')
            name_dict = np.loadtxt(label_path, dtype='str', delimiter='\n')
            for k in range(detection_num):
                class_id = int(reshape_value[k * 7 + 1])
                score = float(reshape_value[k * 7 + 2])
                xmin = max(0, min(reshape_value[k * 7 + 3], img_size[1]))
                xmax = max(0, min(reshape_value[k * 7 + 5], img_size[1]))
                ymin = max(0, min(reshape_value[k * 7 + 4], img_size[0]))
                ymax = max(0, min(reshape_value[k * 7 + 6], img_size[0]))
                xmin = (xmin - (img_size[1] - scale_w) / 2)
                xmax = (xmax - (img_size[1] - scale_w) / 2)
                ymin = (ymin - (img_size[0] - scale_h) / 2)
                ymax = (ymax - (img_size[0] - scale_h) / 2)
                xmin = int(max(0, xmin))
                xmax = int(max(0, xmax))
                ymin = int(max(0, ymin))
                ymax = int(max(0, ymax))
                result = name_dict[class_id]+"," +str(score)+","+str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)
                record.write(result, False)
                if args.save_img:
                    cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), (0, 255, 0))
                    text = name_dict[class_id] + ": " + str(score)
                    text_size, _ = cv2.getTextSize(text, 0, 0.5, 1)
                    cv2.putText(show_img, text, (xmin, ymin + text_size[1]), 0, 0.5, (255, 255, 255), 1)
            if args.save_img:
                print("saving images")
                cv2.imwrite(args.output_dir + "/" + img_name + ".jpg", show_img)
