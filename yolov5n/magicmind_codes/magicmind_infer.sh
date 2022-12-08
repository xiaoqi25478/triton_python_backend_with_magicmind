#!/bin/bash
set -e 

ROOT_PATH=$PWD
QUANT_MODE=force_float32
SHAPE_MUTABLE=true
BATCH=16
IMAGE_NUM=1000

rm -rf $ROOT_PATH/output/*

echo "infer Magicmind model..."
python magicmind_infer.py --magicmind_model $ROOT_PATH/models/mm_model/force_float16_true_1.mm \
                --image_dir /data/datasets/COCO2017/val2017 \
                --image_num ${IMAGE_NUM} \
                --file_list $ROOT_PATH/data/coco_file_list_5000.txt \
                --label_path $ROOT_PATH/data/coco.names \
                --batch ${BATCH} \
                --output_dir $ROOT_PATH/output \

echo "compute acc..."
python $ROOT_PATH/utils/compute_coco_mAP.py --file_list $ROOT_PATH/data/coco_file_list_5000.txt \
                                            --result_dir $ROOT_PATH/output/ \
                                            --ann_dir /data/datasets/COCO2017/ \
                                            --data_type val2017 \
                                            --json_name $ROOT_PATH/output/ \
                                            --output_json $ROOT_PATH/output/output.json \
                                            --image_num $IMAGE_NUM