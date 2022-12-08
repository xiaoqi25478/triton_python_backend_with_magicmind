#!/bin/bash
set -e 

ROOT_PATH=$PWD
BATCH=16
IMAGE_NUM=1000

mkdir -p $ROOT_PATH/output/
for mode in force_float16 force_float32 qint8_mixed_float16
do
    echo "magicmind infer..."
    mkdir -p $ROOT_PATH/output/$mode
    rm -rf $ROOT_PATH/output/$mode/*
    python magicmind_infer.py   --device_id 0 \
                                --magicmind_model $ROOT_PATH/models/mm_model/resnet50_${mode}_true.mm \
                                --image_dir /data/datasets/imagenet/imagenet2 \
                                --image_num $IMAGE_NUM \
                                --name_file $ROOT_PATH/data/imagenet_name.txt \
                                --label_file $ROOT_PATH/data/imagenet_1000.txt \
                                --result_file $ROOT_PATH/output/$mode/infer_result.txt \
                                --result_label_file $ROOT_PATH/output/$mode/eval_labels.txt \
                                --result_top1_file $ROOT_PATH/output/$mode/eval_result_1.txt \
                                --result_top5_file $ROOT_PATH/output/$mode/eval_result_5.txt \
                                --batch_size ${BATCH}
    echo "compute acc..."
    python $ROOT_PATH/utils/compute_top1_and_top5.py \
                    --result_label_file $ROOT_PATH/output/$mode/eval_labels.txt \
                    --result_1_file $ROOT_PATH/output/$mode/eval_result_1.txt \
                    --result_5_file $ROOT_PATH/output/$mode/eval_result_5.txt \
                    --top1andtop5_file $ROOT_PATH/output/$mode/eval_result.txt
done 
