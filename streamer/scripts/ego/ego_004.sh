exp_name='ego_004'

python train.py \
    --dataset data/ego4d  \
    --name "$exp_name"_train \
    --type streamer \
    --dataset_split train \
    --dataset_percent 65 \
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
    --hgn_timescale True \
    --hgn_reach True \
    --bp_up True \
    --bp_down True \
    --optimize True \
    --optimize_every 100 \
    --log_base_every 1000 \
    --evolve_every 50000 \
    --save_every 50000 \
    --log_prefix /data/D2/datasets/ego4d/v2/full_scale/ \
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

