from typing import Optional

from datasets import load_dataset, concatenate_datasets, DatasetDict


def _add_subset_column(dataset_dict: DatasetDict, subset_name: str):
    """
    Add the "subset" column to each split in `dataset_dict` with value `subset_name`.
    """
    result = {}
    for split in dataset_dict:
        dataset = dataset_dict[split]
        dataset = dataset.add_column("subset", [subset_name] * len(dataset))
        result[split] = dataset
    return DatasetDict(result)


def _prep_mrqa(
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
    preprocessing_num_workers: int = 1,
    overwrite_cache: bool = False,
):
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
    mrqa_datasets = mrqa_datasets.remove_columns(["context_tokens", "question_tokens", "answers"])
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
    return mrqa_datasets


def _prep_ropes(
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
    preprocessing_num_workers: int = 1,
    overwrite_cache: bool = False,
):
    # ROPES dataset
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
    ropes_datasets = _add_subset_column(dataset_dict=ropes_datasets, subset_name="ROPES")
    return ropes_datasets


def _prep_squad_v2(
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
):
    # Has train, validation
    squad_v2_datasets = load_dataset(
        "squad_v2",
        None,
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    squad_v2_datasets = _add_subset_column(dataset_dict=squad_v2_datasets, subset_name="SQuADV2")
    return squad_v2_datasets


def _prep_adversarial_qa(
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
):
    # Has train, validation, test
    adversarial_qa_datasets = load_dataset(
        "adversarial_qa",
        "adversarialQA",
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    # Remove extra column
    adversarial_qa_datasets.remove_columns("metadata")
    adversarial_qa_datasets = _add_subset_column(dataset_dict=adversarial_qa_datasets, subset_name="adversarialQA")
    return adversarial_qa_datasets


def _prep_synqa(
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
):
    synqa_datasets = load_dataset(
        "mbartolo/synQA",
        None,
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    synqa_datasets = _add_subset_column(dataset_dict=synqa_datasets, subset_name="synQA")
    return synqa_datasets


# BlendQA
def get_blendqa(
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

    :param cache_dir: Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.
    :param use_auth_token:
    :param preprocessing_num_workers:
    :param overwrite_cache: Whether to overwrite the cached training and evaluation sets.
    """
    # MRQA Dataset
    mrqa_datasets = _prep_mrqa(
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        preprocessing_num_workers=preprocessing_num_workers,
        overwrite_cache=overwrite_cache
    )

    # ROPES dataset
    ropes_datasets = _prep_ropes(
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        preprocessing_num_workers=preprocessing_num_workers,
        overwrite_cache=overwrite_cache
    )

    # SQuADV2 Dataset
    # Has train, validation
    squad_v2_datasets = _prep_squad_v2(cache_dir=cache_dir, use_auth_token=use_auth_token)

    # adversarial_qa dataset
    # Has train, validation, test
    adversarial_qa_datasets = _prep_adversarial_qa(cache_dir=cache_dir, use_auth_token=use_auth_token)

    # Has train
    synqa_datasets = _prep_synqa(cache_dir=cache_dir, use_auth_token=use_auth_token)

    # Construct BlendQA
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
            mrqa_datasets["validation"],
            ropes_datasets["validation"],
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
