set -e 

export PYTHONNOUSERSITE=True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:/usr/local/neuware/lib/llvm-mm/lib:$LD_LIBRARY_PATH

for bs in 1 4 8 16 32 64
do 
    perf_analyzer -m resnet50 --percentile=99 -b $bs
done
