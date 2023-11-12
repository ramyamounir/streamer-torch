exp_name='finegym_001'

python train.py \
    --dataset data/finegym \
    --name "$exp_name"_train \
    --type streamer \
    --dataset_split train \
    --dataset_percent 100 \
    --buffer_size 10 \
    --demarcation_mode average \
    --distance_mode distance \
    --window_size 50 \
    --modifier_type multiply \
    --modifier 1 \
    --optimize_every 10 \
    --average_every 10 \
    --lr 1e-4 \
    --log_base_every 1000 \
    --log_prefix /data/D2/datasets/finegym/videos/ \
    --log_postfix mp4 \
    --init_layers 1 \
    --evolve_every 10000 \
    --hgn_timescale True \
    --hgn_reach True \
    --bp_up True \
    --bp_down True \
    --optimize True \
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
    --dataset data/finegym  \
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
    --buffer_size 10 \
    --lr 1e-4 \
    --log_base_every 1000 \
    --log_prefix /data/D2/datasets/finegym/videos/ \
    --log_postfix mp4 \
    --optimize False \
    --tb \
    --dbg \


# Check the exit code of the first script
if [ $? -eq 0 ]; then
    echo "Testing completed successfully"
else
    echo "Testing failed with exit code $?"
    exit $?
fi

