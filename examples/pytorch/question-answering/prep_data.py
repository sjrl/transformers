from typing import Optional

from datasets import load_dataset, concatenate_datasets, DatasetDict


# QABlend
def get_qablend(
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
    preprocessing_num_workers: int = 1,
    overwrite_cache: bool = False,
):
    """
    Blend multiple QA datasets from HuggingFace
    MRQA:

    Squad V2 (only no answer questions):

    SynQA:

    AdversarialQA:

    ROPES:

    """
    # TODO Alternatively I could download the mrqa datasets repo and modify the mrqa.py script to output the desired format.
    #      This could end up being much faster if this map takes awhile ...
    mrqa_datasets = load_dataset(
        "mrqa",
        None,
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    # Filter out squad datapoints
    mrqa_datasets = mrqa_datasets.filter(lambda example: example["subset"] != "SQuAD")
    # Remove unneeded columns
    mrqa_datasets = mrqa_datasets.remove_columns(["context_tokens", "question_tokens", "answers", "subset"])
    # Rename columns
    mrqa_datasets = mrqa_datasets.rename_column("qid", "id")
    # Map detected_answers to answers in the correct format
    mrqa_datasets.map(
        lambda example: {
            "answers": {
                "text": example["detected_answers"]["text"],
                "answer_start": [x["start"] for x in example["detected_answers"]["char_spans"]]
            }
        },
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=["detected_answers"],
        load_from_cache_file=not overwrite_cache,
        desc="Formatting MRQA",
    )
    ropes_datasets = load_dataset(
        "ropes",
        None,
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    # TODO Check average context length to know how big context length for the model should be.
    # Create context string
    ropes_datasets.map(
        lambda example: {"context": example["background"] + " " + example["situation"]},
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=["background", "situation"],
        load_from_cache_file=not overwrite_cache,
        desc="Creating context for ROPES",
    )
    # TODO Update answers to include answers_start and duplicates if answer appears more than once in the text.
    #      Will probably need a function instead of a lambda function
    ropes_datasets.map(
        lambda example: {"answers": example["answers"]},
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Updating answers for ROPES",
    )
    # Has train, validation
    squad_v2_datasets = load_dataset(
        "squad_v2",
        None,
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    # Has train, validation, test
    adversarial_qa_datasets = load_dataset(
        "adversarial_qa",
        "adversarialQA",
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    # Remove extra column
    adversarial_qa_datasets.remove_columns("metadata")
    # Has train
    synqa_datasets = load_dataset(
        "mbartolo/synQA",
        None,
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    train = concatenate_datasets(
        [
            mrqa_datasets["train"],
            ropes_datasets["train"],
            squad_v2_datasets["train"],
            adversarial_qa_datasets["train"],
            synqa_datasets["train"]
        ]
    )
    validation = concatenate_datasets(
        [
            ropes_datasets["validation"],
            mrqa_datasets["validation"],
            squad_v2_datasets["validation"],
            adversarial_qa_datasets["validation"],
            # synqa_datasets["validation"]  # Doesn't exist
        ]
    )
    test = concatenate_datasets(
        [
            ropes_datasets["test"],
            mrqa_datasets["test"],
            adversarial_qa_datasets["test"],
            # squad_v2_datasets["test"],  # N/A
            # synqa_datasets["test"]  # Doesn't exist
        ]
    )
    return DatasetDict({"train": train, "validation": validation, "test": test})
