#!/bin/bash
set -e 
set -x

echo "基于[yolov5](https://github.com/ultralytics/yolov5.git) v6.1分支"

# 数据集路径
# /data/datasets/COCO2017

pip install -r requirements.txt
ROOT_PATH=$PWD
mkdir -p $ROOT_PATH/codes
mkdir -p $ROOT_PATH/weights
mkdir -p $ROOT_PATH/models/pt_model
mkdir -p $ROOT_PATH/models/mm_model
mkdir -p $ROOT_PATH/output

if [ ! -f $ROOT_PATH/models/pt_model/yolov5n.pt ];then 
    wget -c https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt -O $ROOT_PATH/models/pt_model/yolov5n.pt
else 
    echo "yolov5n.pt is already exited!"
fi

cd $ROOT_PATH/codes
if [ -d "yolov5" ];
then
  echo "yolov5 already exists."
else
  echo "git clone yolov5..."
  git clone https://github.com/ultralytics/yolov5.git
  cd yolov5
  git checkout -b v6.1 v6.1
fi

# patch-yolov5
if grep -q "$ROOT_PATH/models/pt_model/yolov5m_traced.pt" $ROOT_PATH/codes/yolov5/export.py;
then 
  echo "modifying the yolov5m has been already done"
else
  echo "modifying the yolov5m..."
  cd $ROOT_PATH/codes/yolov5/
  git apply $ROOT_PATH/export_model/yolov5_v6_1_pytorch.patch
fi

cd $ROOT_PATH
# patch-torch-cocodataset
if grep -q "SiLU" /usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py;
then
  echo "SiLU activation operator already exists.";
else
  echo "add SiLU op in '/usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py and activation.py'"
  patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/__init__.py < $ROOT_PATH/export_model/init.patch
  patch -p0 /usr/lib/python3.7/site-packages/torch/nn/modules/activation.py < $ROOT_PATH/export_model/activation.patch
fi
