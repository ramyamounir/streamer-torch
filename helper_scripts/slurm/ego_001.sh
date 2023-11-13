exp_name='ego_001'

python train.py \
    --dataset data/ego4d  \
    --name "$exp_name"_train \
    --p_name ego4d \
    --p_device slurm \
    --p_partition Contributors \
    --p_n_nodes 2 \
    --p_n_gpus 4 \
    --p_n_cpus 2 \
    --p_ram 32 \
    --p_backend nccl \
    --dataset_percent 25 \
    \
    --max_layers 3 \
    --evolve_every 50000 \
    --buffer_size 20 \
    --force_fixed_buffer True \
    \
    --demarcation_mode average \
    --distance_mode similarity \
    \
    --lr 1e-4 \
    --alpha 3 \
    --optimize_every 100 \
    --average_every 100 \
    --optimize True \
    --save_every 25000 \
    \
    --dbg \
    --tb \
    --log_prefix /data/D2/datasets/ego4d/videos/ \
    --log_postfix MP4 \
    --log_base_every 1000 \

# Check the exit code of the first script
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $?"
    exit $?
fi



