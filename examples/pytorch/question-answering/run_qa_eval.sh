#!/bin/bash

#pythia-410m
python run_qa.py \
  --model_name_or_path EleutherAI/"$1" \
  --dataset_name squad_v2 \
  --output_dir experiments/"$1"/"$2"/model/ \
  --version_2_with_negative True \
  --max_seq_length 384 \
  --doc_stride 128 \
  --do_eval \
  --per_device_eval_batch_size 8 \
  --tf32 \
  --dataloader_num_workers 6 \
  --preprocessing_num_workers 6 \
  --overwrite_output_dir False > experiments/"$1"/"$2"/eval_squad2.log 2>&1 &


python run_qa.py \
  --model_name_or_path EleutherAI/"$1" \
  --dataset_name squad \
  --output_dir experiments/"$1"/"$2"/model/ \
  --version_2_with_negative False \
  --max_seq_length 384 \
  --doc_stride 128 \
  --do_eval \
  --per_device_eval_batch_size 8 \
  --tf32 \
  --dataloader_num_workers 6 \
  --preprocessing_num_workers 6 \
  --overwrite_output_dir False > experiments/"$1"/"$2"/eval_squad.log 2>&1 &