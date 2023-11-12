exp_name='ego_005'

python train.py \
    --dataset data/ego4d  \
    --name "$exp_name"_train \
    --type streamer \
    --dataset_split train \
    --dataset_percent 25 \
    --buffer_size 10 \
    --demarcation_mode average \
    --loss_threshold 0.1 \
    --distance_mode distance \
    --window_size 50 \
    --modifier_type multiply \
    --modifier 1 \
    --optimize_every 10 \
    --average_every 10 \
    --lr 1e-4 \
    --log_base_every 1000 \
    --log_prefix /data/D2/datasets/ego4d/v2/full_scale/ \
    --log_postfix mp4 \
    --init_layers 1 \
    --evolve_every 50000 \
    --max_layers 3 \
    --save_every 50000 \
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

