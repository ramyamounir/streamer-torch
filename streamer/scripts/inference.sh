exp_name='ego_002'
folder_path="out/${exp_name}_inference"

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --weights $(ls -1 out/"$exp_name"_train/checkpoints/model_*.pth 2>/dev/null | sort -t_ -k2 -n | tail -n 1) \
    --input /data/D2/datasets/ego4d/v2/full_scale/b100157a-b154-4009-a297-f0360acc966b.mp4 \
    --output $folder_path \


# --input /data/D2/datasets/epic_kitchen/videos/P01/P01_13.MP4 \

# Check the exit code of the first script
if [ $? -eq 0 ]; then
    echo "Inference completed successfully"
else
    echo "Inference failed with exit code $?"
    exit $?
fi
