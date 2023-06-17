#!/bin/bash

#--model_name_or_path experiments/"$1"/"$2"/model \
#--output_dir experiments/"$1"/"$2"/eval_squad2/ \
#--overwrite_output_dir False > experiments/"$1"/"$2"/eval_squad2.log 2>&1 &
#
#--model_name_or_path VMware/deberta-v3-large-mrqa \
#--output_dir experiments/vmware-deberta-v3-large-mrqa/eval_squad2/ \
#--overwrite_output_dir False > experiments/vmware-deberta-v3-large-mrqa/eval_squad2.log 2>&1 &
python run_qa.py \
  --model_name_or_path experiments/"$1"/"$2"/model \
  --dataset_name squad_v2 \
  --output_dir experiments/"$1"/"$2"/eval_squad2/ \
  --version_2_with_negative True \
  --max_seq_length 512 \
  --doc_stride 128 \
  --do_eval \
  --per_device_eval_batch_size 24 \
  --tf32 True \
  --dataloader_num_workers 6 \
  --preprocessing_num_workers 6 \
  --bf16_full_eval \
  --eval_accumulation_steps 2 \
  --overwrite_output_dir False > experiments/"$1"/"$2"/eval_squad2.log 2>&1 &

#process_id=$!
#echo "PID: $process_id"
#wait $process_id

#--model_name_or_path experiments/"$1"/"$2"/model \
#--output_dir experiments/"$1"/"$2"/eval_squad/ \
#--overwrite_output_dir False > experiments/"$1"/"$2"/eval_squad.log 2>&1 &
#
#--model_name_or_path VMware/deberta-v3-large-mrqa \
#--output_dir experiments/vmware-deberta-v3-large-mrqa/eval_squad/ \
#--overwrite_output_dir False > experiments/vmware-deberta-v3-large-mrqa/eval_squad.log 2>&1 &
python run_qa.py \
  --model_name_or_path experiments/"$1"/"$2"/model \
  --dataset_name squad \
  --output_dir experiments/"$1"/"$2"/eval_squad/ \
  --version_2_with_negative False \
  --max_seq_length 512 \
  --doc_stride 128 \
  --do_eval \
  --per_device_eval_batch_size 24 \
  --tf32 True \
  --dataloader_num_workers 6 \
  --preprocessing_num_workers 6 \
  --bf16_full_eval \
  --eval_accumulation_steps 2 \
  --overwrite_output_dir False > experiments/"$1"/"$2"/eval_squad.log 2>&1 &
