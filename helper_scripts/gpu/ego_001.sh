exp_name='ego_001'

python train.py \
    --dataset data/ego4d  \
    --name "$exp_name"_train \
    --type streamer \
    --p_device gpu \
    --dataset_split train \
    --dataset_percent 25 \
    --buffer_size 10 \
    --demarcation_mode average \
    --normalize_imgs False \
    --distance_mode similarity \
    --optimize True \
    --optimize_every 100 \
    --average_every 100 \
    --evolve_every 50000 \
    --save_every 25000 \
    --lr 1e-4 \
    --log_base_every 1000 \
    --log_prefix /data/D2/datasets/ego4d/videos/ \
    --log_postfix MP4 \
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



