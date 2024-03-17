#!/bin/bash

export TASK_NAME=mnli

mkdir -p ./experiments/$TASK_NAME/"$1"/"$2"

python run_glue.py \
  --model_name_or_path google/"$1" \
  --output_dir ./experiments/$TASK_NAME/"$1"/"$2"/model \
  --task_name $TASK_NAME \
  --max_seq_length 256 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_step 4 \
  --learning_rate 1e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.10 \
  --num_train_epochs 3 \
  --evaluation_strategy steps \
  --logging_strategy steps \
  --logging_steps 50 \
  --eval_steps 200 \
  --save_steps 400 \
  --save_total_limit 2 \
  --dataloader_num_workers 6 \
  --max_eval_samples 1000 \
  --overwrite_output_dir False \
  --optim adamw_bnb_8bit \
  --tf32 True \
  --bf16 \
  --pad_to_max_length False \
  --gradient_checkpointing True > experiments/$TASK_NAME/"$1"/"$2"/run.log 2>&1 &
  #--preprocessing_num_workers 6 \
