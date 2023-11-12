exp_name='ego_006'

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
    --optimize True \
    --optimize_every 10 \
    --average_every 10 \
    --lr 1e-4 \
    --log_base_every 100 \
    --log_prefix /data/D2/datasets/ego4d/videos/ \
    --log_postfix MP4 \
    --evolve_every 50000 \
    --save_every 25000 \
    --max_layers 3 \
    --tb \
    --dbg \

# Check the exit code of the first script
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $?"
    exit $?
fi



