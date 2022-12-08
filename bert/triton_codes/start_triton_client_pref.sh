set -e 

export PYTHONNOUSERSITE=True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:/usr/local/neuware/lib/llvm-mm/lib:$LD_LIBRARY_PATH

for bs in 1 4 8 16 32 64
do 
    perf_analyzer -m bert_case --percentile=99 -z -b $bs
done
