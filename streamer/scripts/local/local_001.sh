exp_name='local_001'

python train.py \
    --dataset data/epic  \
    --name "$exp_name"_train \
    --type streamer \
    --dataset_split train \
    --dataset_percent 100 \
    --buffer_size 10 \
    --demarcation_mode average \
    --loss_threshold 0.1 \
    --distance_mode distance \
    --window_size 50 \
    --modifier_type multiply \
    --modifier 1 \
    --lr 1e-4 \
    --init_layers 1 \
    --max_layers 3 \
    --optimize True \
    --optimize_every 10 \
    --log_base_every 10 \
    --evolve_every 10000 \
    --save_every 10000 \
    --log_prefix /data/D2/datasets/epic_kitchen/videos/ \
    --log_postfix mp4 \
    --tb \
    --dbg \


# Check the exit code of the first script
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $?"
    exit $?
fi

