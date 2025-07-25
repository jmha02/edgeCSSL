python3 main_pretrain.py \
    --dataset cifar100 \
    --encoder vit_small \
    --data_dir ./data \
    --train_dir train \
    --val_dir val \
    --max_epochs 100 \
    --devices 1 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --lr 0.6 \
    --classifier_lr 0.1 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --min_scale 0.08 \
    --size 32 \
    --num_crops 2 \
    --name dino-vit-small-cifar100 \
    --project cassle \
    --entity cassle_ssl \
    --save_checkpoint \
    --method dino \
    --proj_hidden_dim 2048 \
    --output_dim 256 \
    --num_prototypes 4096 \
    --student_temperature 0.1 \
    --teacher_temperature 0.07 \
    --warmup_teacher_temperature 0.04 \
    --warmup_teacher_temperature_epochs 50 \
    --clip_grad 3.0 \
    --freeze_last_layer 1 \
    --base_tau_momentum 0.996 \
    --final_tau_momentum 1.0 \
    --scheduler cosine \
    --task_idx 0 \
    --split_strategy data \
    --num_tasks 1