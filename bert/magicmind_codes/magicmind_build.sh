set -e
python magicmind_build.py "fp32"
python magicmind_build.py "fp16"
# 暂时无int8
#python magicmind_build_mm.py "int8"
