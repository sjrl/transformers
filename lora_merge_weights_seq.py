import argparse
import os
from peft import LoraConfig, PeftModelForSequenceClassification, PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer


def merge_lora_weights(peft_model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(peft_model_name_or_path)

    lora_config = LoraConfig.from_pretrained(peft_model_name_or_path)
    config = AutoConfig.from_pretrained(peft_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        lora_config.base_model_name_or_path,
        config=config,
    )
    model = PeftModel.from_pretrained(
        model=model, model_id=peft_model_name_or_path, is_trainable=False
    )
    # model.base_model is the LoraModel
    # model.base_model.merge_and_unload() returns LoraModel.model which is the underlying transformers model
    merged_model = model.merge_and_unload()

    save_dir = os.path.join(peft_model_name_or_path, "merged_model")
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    merged_model.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--peft_model_name_or_path', type=str, default="./", nargs="?")
    args = parser.parse_args()

    merge_lora_weights(peft_model_name_or_path=args.peft_model_name_or_path)

