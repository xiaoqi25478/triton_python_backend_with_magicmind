set -e

# 1.对原始的pt模型进行jit
ROOT_PATH=$PWD
cd $ROOT_PATH/codes/yolov5
echo "jit model begin..."
python $ROOT_PATH/codes/yolov5/export.py --weights $ROOT_PATH/models/pt_model/yolov5n.pt --imgsz 640 640 --include torchscript --batch-size 1
echo "jit model end..."

# 2.对jit过后的torchscript模型进行build
# INPUT -1,640,640,3 UINT8
# OUTPUT -1,1000,7 -1  FLOAT32 INT32

cd $ROOT_PATH/gen_model
QUANT_MODE=force_float32 
SHAPE_MUTABLE=true 
BATCH_SIZE=1
CONF_THRES=0.001
IOU_THRES=0.65 
MAX_DET=1000

echo "build model begin..."
for quant_mode in force_float32 force_float16 qint8_mixed_float16
do
python $ROOT_PATH/gen_model/gen_model.py    --pt_model $ROOT_PATH/models/pt_model/yolov5n.torchscript \
                                            --output_model $ROOT_PATH/models/mm_model/${quant_mode}_${SHAPE_MUTABLE}_${BATCH_SIZE}.mm \
                                            --image_dir /data/datasets/COCO2017/val2017 \
                                            --quant_mode ${QUANT_MODE} \
                                            --shape_mutable ${SHAPE_MUTABLE} \
                                            --batch_size ${BATCH_SIZE} \
                                            --conf_thres ${CONF_THRES} \
                                            --iou_thres ${IOU_THRES} \
                                            --max_det ${MAX_DET}
done
echo "build model end..."