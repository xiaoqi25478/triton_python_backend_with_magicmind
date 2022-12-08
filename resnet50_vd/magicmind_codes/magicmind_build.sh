set -e

python magicmind_build.py --quant_mode qint8_mixed_float16  --shape_mutable "true"  --onnx_model ./models/onnx_model/resnet50_vd.onnx 
python magicmind_build.py --quant_mode force_float32        --shape_mutable "true"  --onnx_model ./models/onnx_model/resnet50_vd.onnx
python magicmind_build.py --quant_mode force_float16        --shape_mutable "true"  --onnx_model ./models/onnx_model/resnet50_vd.onnx
