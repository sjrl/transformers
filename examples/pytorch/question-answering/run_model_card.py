import os
import json
from argparse import ArgumentParser
import torch
from typing import Optional


def get_dataset_metrics(
    metrics_path: str,
    dataset_name: str,
    dataset_type: str,
    dataset_config: str,
    dataset_split: str
) -> str:
    with open(metrics_path) as f1:
        metrics = json.load(f1)
    eval_exact = metrics.get("eval_exact")
    if eval_exact is None:
        eval_exact = metrics["eval_exact_match"]
    metrics_readme = f"""- task:
      type: question-answering
      name: Question Answering
    dataset:
      name: {dataset_name}
      type: {dataset_type}
      config: {dataset_config}
      split: {dataset_split}
    metrics:
    - type: exact_match
      value: {eval_exact:.3f}
      name: Exact Match
    - type: f1
      value: {metrics["eval_f1"]:.3f}
      name: F1"""
    return metrics_readme


def get_training_procedure(training_args_file: str) -> str:
    if not os.path.isfile(training_args_file):
        return "TRAINING_PROCEDURE"

    training_args = torch.load(training_args_file)
    try:
        warmup_ratio = training_args.lr_scheduler_warmup_ratio
    except AttributeError:
        warmup_ratio = training_args.warmup_ratio
    template = f"""## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: {training_args.learning_rate:.0E}
- train_batch_size: {training_args.train_batch_size}
- eval_batch_size: {training_args.eval_batch_size}
- seed: {training_args.seed}
- gradient_accumulation_steps: {training_args.gradient_accumulation_steps}
- total_train_batch_size: {training_args.train_batch_size * training_args.gradient_accumulation_steps}
- optimizer: Adam with betas=({training_args.adam_beta1:.1f},{training_args.adam_beta2:.3f}) and epsilon={training_args.adam_epsilon:.0E}
- lr_scheduler_type: {training_args.lr_scheduler_type}
- lr_scheduler_warmup_ratio: {warmup_ratio:.1f}
- num_epochs: {training_args.num_train_epochs:.1f}"""
    return template


def create_model_card(model_path: str, output_path: Optional[str] = None) -> None:
    config_file = os.path.join(model_path, "config.json")
    if os.path.isfile(config_file):
        with open(os.path.join(model_path, "config.json")) as f1:
            config = json.load(f1)
        base_model = config["_name_or_path"]
        model_name = f"deepset/{base_model.split('/')[-1]}"
    else:
        base_model = "BASE_MODEL"
        model_name = "MODEL_NAME"

    training_procedure = get_training_procedure(training_args_file=os.path.join(model_path, "training_args.bin"))

    squad_v2 = get_dataset_metrics(
        metrics_path=os.path.join(model_path, "eval_squad_v2", "all_results.json"),
        dataset_name="squad_v2",
        dataset_type="squad_v2",
        dataset_config="squad_v2",
        dataset_split="validation",
    )
    squad = get_dataset_metrics(
        metrics_path=os.path.join(model_path, "eval_squad", "all_results.json"),
        dataset_name="squad",
        dataset_type="squad",
        dataset_config="plain_text",
        dataset_split="validation",
    )
    adversarial_qa = get_dataset_metrics(
        metrics_path=os.path.join(model_path, "eval_adversarial_qa_adversarialQA", "all_results.json"),
        dataset_name="adversarial_qa",
        dataset_type="adversarial_qa",
        dataset_config="adversarialQA",
        dataset_split="validation",
    )
    squad_adversarial = get_dataset_metrics(
        metrics_path=os.path.join(model_path, "eval_squad_adversarial_AddOneSent", "all_results.json"),
        dataset_name="squad_adversarial",
        dataset_type="squad_adversarial",
        dataset_config="AddOneSent",
        dataset_split="validation",
    )
    squadshifts_amazon = get_dataset_metrics(
        metrics_path=os.path.join(model_path, "eval_squadshifts_amazon", "all_results.json"),
        dataset_name="squadshifts amazon",
        dataset_type="squadshifts",
        dataset_config="amazon",
        dataset_split="test",
    )
    squadshifts_new_wiki = get_dataset_metrics(
        metrics_path=os.path.join(model_path, "eval_squadshifts_new_wiki", "all_results.json"),
        dataset_name="squadshifts new_wiki",
        dataset_type="squadshifts",
        dataset_config="new_wiki",
        dataset_split="test",
    )
    squadshifts_nyt = get_dataset_metrics(
        metrics_path=os.path.join(model_path, "eval_squadshifts_nyt", "all_results.json"),
        dataset_name="squadshifts nyt",
        dataset_type="squadshifts",
        dataset_config="nyt",
        dataset_split="test",
    )
    squadshifts_reddit = get_dataset_metrics(
        metrics_path=os.path.join(model_path, "eval_squadshifts_reddit", "all_results.json"),
        dataset_name="squadshifts reddit",
        dataset_type="squadshifts",
        dataset_config="reddit",
        dataset_split="test",
    )

    template = f"""---
language:
- en
license: cc-by-4.0
library_name: transformers
tags:
- question-answering
- squad
- squad_v2
datasets:
- squad_v2
- squad
base_model: {base_model}
model-index:
- name: deepset/{model_name}
  results:
  {squad_v2}
  {squad}
  {adversarial_qa}
  {squad_adversarial}
  {squadshifts_amazon}
  {squadshifts_new_wiki}
  {squadshifts_nyt}
  {squadshifts_reddit}
---

# {model_name} for Extractive QA

This is the [{base_model}](https://huggingface.co/{base_model}) model, fine-tuned using the [SQuAD2.0](https://huggingface.co/datasets/squad_v2) dataset. It's been trained on question-answer pairs, including unanswerable questions, for the task of Extractive Question Answering.

## Overview
**Language model:** {base_model}  
**Language:** English  
**Downstream-task:** Extractive QA  
**Training data:** SQuAD 2.0  
**Eval data:** SQuAD 2.0  
**Infrastructure**: 1x NVIDIA A10G  

## Model Usage

```python
import torch
from transformers import(
  AutoModelForQuestionAnswering,
  AutoTokenizer,
  pipeline
)
model_name = "deepset/{model_name}"

# a) Using pipelines
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
qa_input = {{
'question': 'Where do I live?',
'context': 'My name is Sarah and I live in London'
}}
res = nlp(qa_input)
# {{'score': 0.984, 'start': 30, 'end': 37, 'answer': ' London'}}

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

question = 'Where do I live?'
context = 'My name is Sarah and I live in London'
encoding = tokenizer(question, context, return_tensors="pt")
start_scores, end_scores = model(
  encoding["input_ids"],
  attention_mask=encoding["attention_mask"],
  return_dict=False
)

all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
answer_tokens = all_tokens[torch.argmax(start_scores):torch.argmax(end_scores) + 1]
answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
# 'London'
```

## Metrics

```bash
# Squad v2
{{
"eval_HasAns_exact": 84.83468286099865,
    "eval_HasAns_f1": 90.48374860633226,
    "eval_HasAns_total": 5928,
    "eval_NoAns_exact": 91.0681244743482,
    "eval_NoAns_f1": 91.0681244743482,
    "eval_NoAns_total": 5945,
    "eval_best_exact": 87.95586625115808,
    "eval_best_exact_thresh": 0.0,
    "eval_best_f1": 90.77635490089573,
    "eval_best_f1_thresh": 0.0,
    "eval_exact": 87.95586625115808,
    "eval_f1": 90.77635490089592,
    "eval_runtime": 623.1333,
    "eval_samples": 11951,
    "eval_samples_per_second": 19.179,
    "eval_steps_per_second": 0.799,
    "eval_total": 11873
}}

# Squad
{{
"eval_exact_match": 89.29044465468307,
    "eval_f1": 94.9846365606959,
    "eval_runtime": 553.7132,
    "eval_samples": 10618,
    "eval_samples_per_second": 19.176,
    "eval_steps_per_second": 0.8
}}
```

{training_procedure}

### Framework versions

- Transformers 4.30.0.dev0
- Pytorch 2.0.1+cu117
- Datasets 2.12.0
- Tokenizers 0.13.3
"""

    with open(os.path.join(output_path, "README.md"), "w") as f1:
        f1.write(template)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest="model_path", required=True)
    parser.add_argument('-o', '--output', dest="output_path")
    args = parser.parse_args()

    base_output_path = args.output_path
    if base_output_path is None:
        base_output_path = args.model_path

    create_model_card(model_path=args.model_path, output_path=base_output_path)
