from huggingface_hub import HfApi
import argparse
from pathlib import Path
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-id', dest="repo", type=str, default="./", required=True)
    parser.add_argument('--folder', type=str, default="./", nargs="?")
    args = parser.parse_args()

    api = HfApi()
    potential_files = [
        # Model
        "config.json",
        "modeling_t5qa.py",
        "modeling_t5seq.py",
        "pytorch_model.bin",
        "model.safetensors",
        # Tokenizer
        "spiece.model",  # not always present
        "added_tokens.json",  # not always present
        "special_tokens_map.json",
        "spm.model",  # not always present
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",  # not always present
    ]
    lora_potential_files = [
        # LoRA model
        "adapter_config.json",
        "adapter_model.bin",
    ]
    lora_potential_files = [os.path.join(args.folder, x) for x in lora_potential_files]
    present_lora_files = [x for x in lora_potential_files if Path(x).is_file()]
    if len(present_lora_files) > 0:
        potential_files = [os.path.join("merged_model", x) for x in potential_files]
    potential_files = [os.path.join(args.folder, x) for x in potential_files]
    present_files = [x for x in potential_files if Path(x).is_file()]
    present_files.extend(present_lora_files)
    if len(present_files) == 0:
        raise RuntimeError(f"No files to upload were found")
    for f in present_files:
        path_in_repo = f.split(args.folder)[-1]
        if "merged_model" in path_in_repo:
            path_in_repo = path_in_repo.split("merged_model/")[-1]
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=path_in_repo,
            repo_id=args.repo,
        )
