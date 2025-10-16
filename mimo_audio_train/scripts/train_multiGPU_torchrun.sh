#!/bin/bash

echo "=== MiMo-Audio Multi-GPU Training ==="


# 1. Base environments
export CUDA_VISIBLE_DEVICES=7

# 2. Distributed Training Config
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export OMP_NUM_THREADS=1

# NCCL Config
export NCCL_DEBUG=INFO

export NCCL_IB_DISABLE=1
export NCCL_IBEXT_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET=Socket

# Disable P2P
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=0

export NCCL_SOCKET_IFNAME=eth0

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 3. Model and Data Path
MODEL_PATH="XiaomiMiMo/MiMo-Audio-7B-Instruct"
SPEECH_TOKENIZER_PATH="XiaomiMiMo/MiMo-Audio-Tokenizer"
DATA_PATH="data_list/example_train.json"
VALID_PATH="data_list/example_validate.json"
OUTPUT_DIR="./checkpoints/mimo_audio"

# 4. Output Dir
mkdir -p $OUTPUT_DIR
mkdir -p ./logs

# 5. GPU Numbers
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "Training Configuration:"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  GPU Numbers: $NUM_GPUS"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Communication Backend: NCCL (GPU Communication - High Performance)"

# 6. Begin Training
echo "Begin Training..."
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train.py \
    --model_name_or_path $MODEL_PATH \
    --speech_tokenizer_name_or_path $SPEECH_TOKENIZER_PATH \
    --data_path $DATA_PATH \
    --validate_path $VALID_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --save_total_limit 3 \
    --eval_strategy "steps" \
    --bf16 True \
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --report_to tensorboard \
    --logging_first_step True \
    --log_level info \
    --disable_tqdm False \
    --lora_enable True \
    --ddp_backend nccl \
    --ddp_timeout 3600 \
    --ddp_find_unused_parameters True \
    2>&1 | tee ./logs/mimo_audio_nccl.log

# 7. Finished
if [ $? -eq 0 ]; then
    echo "✅ NCCL Training Finished！"
    echo "📁 Ckpt saved at: $OUTPUT_DIR"
else
    echo "❌ NCCL Training Error，Please Check ./logs/mimo_audio_nccl.log"
fi
