#!/bin/bash

export TASK_NAME=mnli

python run_glue.py \
  --model_name_or_path "$1" \
  --output_dir $1/"eval_${TASK_NAME}"/ \
  --task_name $TASK_NAME \
  --max_seq_length 256 \
  --do_eval \
  --per_device_eval_batch_size 16 \
  --dataloader_num_workers 6 \
  --overwrite_output_dir False \
  --tf32 True \
  --eval_accumulation_steps 2 \
  --pad_to_max_length False > $1/"${TASK_NAME}.log" 2>&1 &

#  --bf16_full_eval \
