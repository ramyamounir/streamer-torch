ckpt_name='ego_001'
exp_name='epic_007'

python train.py \
    --dataset data/epic  \
    --name "$exp_name"_test \
    --type streamer \
    --init_ckpt $(ls -1 out/"$ckpt_name"_train/checkpoints/model_*.pth 2>/dev/null | sort -t_ -k2 -n | tail -n 1) \
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

