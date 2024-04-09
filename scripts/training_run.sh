counter=1
for config in  4-76.json 6-112 6-204
do
    CUDA_VISIBLE_DEVICES=$counter CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/run_training.py --config scripts/$config & > $config.log
    counter=$((counter+1))
done
