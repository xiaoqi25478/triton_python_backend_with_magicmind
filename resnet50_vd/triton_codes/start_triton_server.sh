set -e 

export PYTHONNOUSERSITE=True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:/usr/local/neuware/lib/llvm-mm/lib:$LD_LIBRARY_PATH
source /root/miniconda3/bin/activate
conda activate python37_mm0.13.0

tritonserver --model-repository `pwd`/mm_models