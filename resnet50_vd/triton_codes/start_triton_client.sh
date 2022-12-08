set -e 
ROOT_PATH=$PWD
mkdir -p $ROOT_PATH/output
rm -rf $ROOT_PATH/output/*

echo "infer Magicmind model..."
python mm_models/resnet50/client.py

echo "compute acc..."
python $ROOT_PATH/utils/compute_top1_and_top5.py \
                --result_label_file $ROOT_PATH/output/eval_labels.txt \
                --result_1_file $ROOT_PATH/output/eval_result_1.txt \
                --result_5_file $ROOT_PATH/output/eval_result_5.txt \
                --top1andtop5_file $ROOT_PATH/output/eval_result.txt