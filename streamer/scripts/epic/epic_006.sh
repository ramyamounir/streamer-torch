ckpt_name='ego_001'
exp_name='epic_006'

python train.py \
    --dataset data/epic  \
    --name "$exp_name"_train \
    --type streamer \
    --init_ckpt $(ls -1 out/"$exp_name"_train/checkpoints/model_*.pth 2>/dev/null | sort -t_ -k2 -n | tail -n 1) \
    --dataset_split train \
    --dataset_percent 100 \
    --buffer_size 10 \
    --demarcation_mode average \
    --loss_threshold 0.1 \
    --distance_mode distance \
    --window_size 50 \
    --modifier_type multiply \
    --modifier 1 \
    --optimize_every 10 \
    --lr 1e-4 \
    --log_base_every 100 \
    --log_prefix /data/D2/datasets/epic_kitchen/videos/ \
    --log_postfix MP4 \
    --optimize_every 10 \
    --evolve_every 10000 \
    --max_layers 3 \
    --force_base_dist \
    --optimize \
    --tb \
    --dbg \

# Check the exit code of the first script
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $?"
    exit $?
fi


python train.py \
    --dataset data/epic  \
    --name "$exp_name"_test \
    --type streamer \
    --init_ckpt $(ls -1 out/"$exp_name"_train/checkpoints/model_*.pth 2>/dev/null | sort -t_ -k2 -n | tail -n 1) \
    --dataset_split test \
    --dataset_percent 100 \
    --buffer_size 10 \
    --demarcation_mode average \
    --loss_threshold 0.1 \
    --distance_mode distance \
    --window_size 50 \
    --modifier_type multiply \
    --modifier 1 \
    --optimize_every 1 \
    --buffer_size 10 \
    --lr 1e-4 \
    --log_base_every 100 \
    --log_prefix /data/D2/datasets/epic_kitchen/videos/ \
    --optimize_every 1 \
    --force_base_dist \
    --tb \
    --dbg \


# Check the exit code of the first script
if [ $? -eq 0 ]; then
    echo "Testing completed successfully"
else
    echo "Testing failed with exit code $?"
    exit $?
fi

