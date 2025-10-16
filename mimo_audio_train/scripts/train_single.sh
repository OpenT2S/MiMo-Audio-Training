#!/bin/bash

# export environment variables
export CUDA_VISIBLE_DEVICES=0  # only use GPU 0

# model and data path
MODEL_PATH="XiaomiMiMo/MiMo-Audio-7B-Instruct"
SPEECH_TOKENIZER_PATH="XiaomiMiMo/MiMo-Audio-Tokenizer"
DATA_PATH="data_list/example_train.json"
VALID_PATH="data_list/example_validate.json"
OUTPUT_DIR="./checkpoints/mimo_audio"

# create output directory
mkdir -p $OUTPUT_DIR
mkdir -p ./logs

echo "start single card training..."
echo "using GPU: $CUDA_VISIBLE_DEVICES"
echo "output directory: $OUTPUT_DIR"

# single card training - not use DeepSpeed self.tokenizer.decode(input_ids[0][0][::self.group_size].cpu().numpy())
python train.py \
    --model_name_or_path $MODEL_PATH \
    --speech_tokenizer_name_or_path $SPEECH_TOKENIZER_PATH \
    --data_path $DATA_PATH \
    --validate_path $VALID_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
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
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --report_to tensorboard \
    --logging_first_step True \
    --log_level info \
    --disable_tqdm False \
    --lora_enable True \
    2>&1 | tee ./logs/mimo_audio_single.log

echo "training completed! checkpoints saved in: $OUTPUT_DIR"
echo "log file saved in: ./logs/mimo_audio_single.log"
