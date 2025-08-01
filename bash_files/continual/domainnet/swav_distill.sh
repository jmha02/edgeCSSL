python3 main_continual.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR/domainnet \
    --split_strategy domain \
    --max_epochs 200 \
    --num_tasks 6 \
    --task_idx 1 \
    --devices 0,1,2,3 \
    --accelerator ddp \
    --sync_batchnorm \
    --num_workers 5 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.8 \
    --min_lr 0.0006 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size 64 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --dali \
    --name swav-domainnet-knowlegde \
    --entity unitn-mhug \
    --project ever-learn \
    --wandb \
    --save_checkpoint \
    --method swav \
    --proj_hidden_dim 2048 \
    --queue_size 3840 \
    --output_dim 128 \
    --num_prototypes 3000 \
    --epoch_queue_starts 100 \
    --freeze_prototypes_epochs 5 \
    --check_val_every_n_epoch 9999 \
    --disable_knn_eval \
    --distiller knowledge \
    --pretrained_model $PRETRAINED_PATH
