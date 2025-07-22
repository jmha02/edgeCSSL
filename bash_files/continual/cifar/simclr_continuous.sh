#!/bin/bash

# Continuous Self-Supervised Learning with SimCLR on CIFAR-100
# This script trains SimCLR in a continual learning setting across multiple tasks

# Set environment variables
export DATA_DIR=${DATA_DIR:-./data}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Configuration
DATASET="cifar100"
METHOD="simclr"
ENCODER="resnet50"
NUM_TASKS=5
SPLIT_STRATEGY="class"  # Split classes across tasks
BATCH_SIZE=64
LEARNING_RATE=0.3
WEIGHT_DECAY=1e-4
MAX_EPOCHS=100
OUTPUT_DIM=128
PROJ_HIDDEN_DIM=2048
TEMPERATURE=0.1

# Augmentation parameters
BRIGHTNESS="0.4"
CONTRAST="0.4" 
SATURATION="0.4"
HUE="0.1"
GAUSSIAN_PROB="0.5"
SOLARIZATION_PROB="0.2"

# Create base experiment directory
EXPERIMENT_DIR="experiments/continual_simclr_$(date +%Y%m%d_%H%M%S)"
mkdir -p $EXPERIMENT_DIR

echo "🚀 Starting Continuous Self-Supervised Learning with SimCLR"
echo "Dataset: $DATASET"
echo "Tasks: $NUM_TASKS"
echo "Split Strategy: $SPLIT_STRATEGY"
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "=================================================="

# Train each task sequentially
for task_idx in $(seq 0 $((NUM_TASKS-1))); do
    echo ""
    echo "🎯 Training Task $task_idx/$((NUM_TASKS-1))"
    echo "Current time: $(date)"
    
    # Set task-specific parameters
    TASK_NAME="${METHOD}-${DATASET}-task${task_idx}"
    CHECKPOINT_DIR="${EXPERIMENT_DIR}/task_${task_idx}"
    
    # For tasks after the first, use previous task's checkpoint
    PRETRAINED_MODEL=""
    if [ $task_idx -gt 0 ]; then
        PREV_TASK=$((task_idx-1))
        PRETRAINED_MODEL="--pretrained_model ${EXPERIMENT_DIR}/task_${PREV_TASK}/last.ckpt"
        echo "📂 Loading from previous task checkpoint: task_${PREV_TASK}"
    fi
    
    # Run continual training
    python main_continual.py \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --method $METHOD \
        --encoder $ENCODER \
        --num_tasks $NUM_TASKS \
        --task_idx $task_idx \
        --split_strategy $SPLIT_STRATEGY \
        --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --max_epochs $MAX_EPOCHS \
        --optimizer sgd \
        --scheduler cosine \
        --warmup_epochs 10 \
        --brightness $BRIGHTNESS \
        --contrast $CONTRAST \
        --saturation $SATURATION \
        --hue $HUE \
        --gaussian_prob $GAUSSIAN_PROB \
        --solarization_prob $SOLARIZATION_PROB \
        --output_dim $OUTPUT_DIM \
        --proj_hidden_dim $PROJ_HIDDEN_DIM \
        --temperature $TEMPERATURE \
        --gpus 1 \
        --precision 16 \
        --num_workers 8 \
        --name $TASK_NAME \
        --save_checkpoint \
        --checkpoint_dir $CHECKPOINT_DIR \
        --checkpoint_frequency 1 \
        --enable_progress_bar \
        --disable_knn_eval \
        $PRETRAINED_MODEL
    
    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo "✅ Task $task_idx completed successfully"
        
        # Save task summary
        echo "Task $task_idx completed at $(date)" >> $EXPERIMENT_DIR/training_log.txt
        echo "Checkpoint saved to: $CHECKPOINT_DIR/last.ckpt" >> $EXPERIMENT_DIR/training_log.txt
    else
        echo "❌ Task $task_idx failed!"
        echo "Task $task_idx FAILED at $(date)" >> $EXPERIMENT_DIR/training_log.txt
        exit 1
    fi
    
    echo "=================================================="
done

echo ""
echo "🎉 Continuous Self-Supervised Learning Complete!"
echo "📊 Training Summary:"
echo "- Total Tasks: $NUM_TASKS"
echo "- Method: $METHOD"
echo "- Dataset: $DATASET"
echo "- Experiment Directory: $EXPERIMENT_DIR"
echo ""
echo "📁 Checkpoints saved in:"
for task_idx in $(seq 0 $((NUM_TASKS-1))); do
    echo "  Task $task_idx: $EXPERIMENT_DIR/task_${task_idx}/last.ckpt"
done
echo ""
echo "🔍 To evaluate the continual learning performance:"
echo "python main_linear.py --pretrained_model $EXPERIMENT_DIR/task_$((NUM_TASKS-1))/last.ckpt --dataset $DATASET"
echo ""
echo "Completed at: $(date)"