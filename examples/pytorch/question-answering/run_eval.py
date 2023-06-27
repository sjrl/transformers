import os
from argparse import ArgumentParser
import gc
import torch

from run_qa import main


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest="model_path", required=True)
    parser.add_argument('-b', '--batch-size', dest="batch_size", default=24)
    args = parser.parse_args()

    datasets = {
        "squad_v2": [None],
        "squad": [None],
        "squadshifts": ["amazon", "new_wiki", "nyt", "reddit"],
        "adversarial_qa": ["adversarialQA"],
        "squad_adversarial": ["AddOneSent"],
        "mrqa": [None],
    }

    for dataset, configs in datasets.items():
        for config in configs:
            gc.collect()
            torch.cuda.empty_cache()
            if config is not None:
                output_dir = os.path.join(args.model_path, f"eval_{dataset}_{config}")
                extra_args = ["--dataset_config_name", config]
            else:
                output_dir = os.path.join(args.model_path, f"eval_{dataset}")
                extra_args = []
            raw_args = [
                "--model_name_or_path", args.model_path,
                "--dataset_name", dataset,
                "--output_dir", output_dir,
                "--version_2_with_negative", "True",
                "--max_seq_length", "512",
                "--doc_stride", "128",
                "--do_eval",
                "--per_device_eval_batch_size", str(args.batch_size),
                "--bf16_full_eval",
                "--tf32", "True",
                "--dataloader_num_workers", "6",
                "--preprocessing_num_workers", "6",
                "--eval_accumulation_steps", "2",
                "--overwrite_output_dir", "False",
            ] + extra_args
            main(raw_args=raw_args)
