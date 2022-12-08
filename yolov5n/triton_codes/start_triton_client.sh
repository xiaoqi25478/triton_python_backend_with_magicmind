set -e 
ROOT_PATH=$PWD
mkdir -p $ROOT_PATH/output
rm -rf $ROOT_PATH/output/*

echo "infer Magicmind model..."
python mm_models/yolov5n/client.py

echo "compute acc..."
python $ROOT_PATH/utils/compute_coco_mAP.py --file_list $ROOT_PATH/data/coco_file_list_5000.txt \
                                            --result_dir $ROOT_PATH/output/ \
                                            --ann_dir /data/datasets/COCO2017/ \
                                            --data_type val2017 \
                                            --json_name $ROOT_PATH/output/ \
                                            --output_json $ROOT_PATH/output/output.json \
                                            --image_num 1000
                                            