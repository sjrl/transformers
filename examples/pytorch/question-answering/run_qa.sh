#!/bin/bash

# --model_name_or_path EleutherAI/pythia-410m \
# --model_name_or_path EleutherAI/pythia-160m \
# --model_name_or_path facebook/opt-350m \
# --model_name_or_path facebook/opt-125m \
# --model_name_or_path bigscience/bloomz-560m \
mkdir -p ./experiments/"$1"/"$2"
#nohup python run_qa.py \
python run_qa.py \
  --model_name_or_path google/"$1" \
  --dataset_name squad_v2 \
  --output_dir experiments/"$1"/"$2"/model/ \
  --version_2_with_negative True \
  --max_seq_length 512 \
  --doc_stride 128 \
  --max_answer_length 100 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.10 \
  --num_train_epochs 1 \
  --evaluation_strategy steps \
  --logging_strategy steps \
  --logging_steps 2 \
  --eval_steps 200 \
  --save_steps 400 \
  --save_total_limit 2 \
  --dataloader_num_workers 6 \
  --preprocessing_num_workers 6 \
  --max_eval_samples 1000 \
  --overwrite_output_dir False \
  --optim adamw_bnb_8bit \
  --fp16 \
  --pad_to_max_length True \
  --gradient_checkpointing True
  #--gradient_checkpointing True > experiments/"$1"/"$2"/run.log 2>&1 &
  #--torch_dtype float16 \
  #--tf32 True \
  #--bf16 \
  #--fp16 \
  # --metric_for_best_model eval_f1 \
  # --greater_is_better True \
  #  --fp16 \
  # Disabled torch compile for now since it is super buggy. E.g. produces wrong math and is not compatible with gradient checkpointing.
  # "--torch_compile", "True",  # w/ 8bit; w/ dynamo: Went from 11.9 GB to 12.4 GB for bloom, speed increased to ~0.55 it/s (1.38 times faster) Would take about 11 hours.
                                # w/ 8bit; w/o dynamo: Went from 11.9 GB to 11.9 GB for bloom, speed is 0.40 it/s. Would take about 15.5 hours.

# Notes:
# bigscience/bloomz-560m might need to be loaded into bf16 or fp16 instead of fp32. This could make it trainable
# --tf32 Use when training on 3070. Can lead to much higher throughput.
# --optim adamw_bnb_8bit Went from 13.5 GB to 11.9 GB for bloom, speed seems roughly the same 0.40 it/s. Would take about 15.5 hours.
# --gradient_checkpointing True
#   w/ 8bit; w/o torch compile: Went from 11.9 GB to 9.6 GB, speed decreased to ~0.30 it/s Would take about 20 hours.
#   w/ 8bit; w/o torch compile; w/o fp16: Went from 11.9 GB to 9.7 GB, speed decreased to ~0.13 it/s Would take about 50 hours.
# --fp16 Turning this off when using 8bit optim and checkpointing led to no mem change.
