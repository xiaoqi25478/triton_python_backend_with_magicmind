#!/bin/bash
set -e 
set -x

# 数据集路径
# /data/datasets/COCO2017
pip install -r requirements.txt
ROOT_PATH=$PWD
mkdir -p $ROOT_PATH/codes
mkdir -p $ROOT_PATH/weights
mkdir -p $ROOT_PATH/models/pt_model
mkdir -p $ROOT_PATH/models/onnx_model
mkdir -p $ROOT_PATH/output
mkdir -p $ROOT_PATH/weights
