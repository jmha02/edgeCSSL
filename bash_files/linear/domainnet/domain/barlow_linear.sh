# all
python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR/domainnet \
    --split_strategy domain \
    --max_epochs 100 \
    --devices 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name barlow-domain_all-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project ever-learn \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint

# quickdraw
python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR/domainnet \
    --split_strategy domain \
    --domain quickdraw \
    --max_epochs 100 \
    --devices 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name barlow-domain_quickdraw-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project ever-learn \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint

# clipart
python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR/domainnet \
    --split_strategy domain \
    --domain clipart \
    --max_epochs 100 \
    --devices 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name barlow-domain_clipart-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project ever-learn \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint

# infograph
python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR/domainnet \
    --split_strategy domain \
    --domain infograph \
    --max_epochs 100 \
    --devices 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name barlow-domain_infograph-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project ever-learn \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint

# painting
python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR/domainnet \
    --split_strategy domain \
    --domain painting \
    --max_epochs 100 \
    --devices 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name barlow-domain_painting-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project ever-learn \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint

# real
python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR/domainnet \
    --split_strategy domain \
    --domain real \
    --max_epochs 100 \
    --devices 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name barlow-domain_real-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project ever-learn \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint

# sketch
python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR/domainnet \
    --split_strategy domain \
    --domain sketch \
    --max_epochs 100 \
    --devices 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name barlow-domain_sketch-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project ever-learn \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint

