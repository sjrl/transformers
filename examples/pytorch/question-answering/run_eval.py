import os
from argparse import ArgumentParser
from run_qa import main


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest="model_path", required=True)
    parser.add_argument('-b', '--batch-size', dest="batch_size", default=24)
    args = parser.parse_args()

    datasets = {
        "squad_v2": None,
        "squad": None,
        "squadshifts": ["amazon", "new_wiki", "nyt", "reddit"],
        "adversarial_qa": ["adversarialQA"],
        "squad_adversarial": ["AddOneSent"],
    }

    for d, c in datasets.items():
        if c is not None:
            output_dir = os.path.join(args.model_path, f"eval_{d}_{c}")
        else:
            output_dir = os.path.join(args.model_path, f"eval_{d}")
        raw_args = [
            "--model_name_or_path", args.model_path,
            "--dataset_name", d,
            "--dataset_config_name", c,
            "--output_dir", output_dir,
            "--version_2_with_negative", "True",
            "--max_seq_length", "512",
            "--doc_stride", "128",
            "--do_eval",
            "--per_device_eval_batch_size", args.batch_size,
            "--bf16_full_eval",
            "--tf32", "True",
            "--dataloader_num_workers", "6",
            "--preprocessing_num_workers", "6",
            "--eval_accumulation_steps", "2",
            "--overwrite_output_dir", "False",
        ]
        main(raw_args=raw_args)
