import re
import string
from typing import Optional, List

from datasets import load_dataset, concatenate_datasets, DatasetDict


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        # Handle "-" symbol separately
        tmp = text.replace("-", " ")
        exclude = set(string.punctuation)
        return "".join(ch for ch in tmp if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def update_answer_start(
    context: str,
    text: List[str],
    answer_start: List[int],
    answer_end: List[int],
):
    # Wrong start
    updated_text = []
    updated_answer_start = []
    updated_answer_end = []
    for old_start, old_end, t in zip(answer_start, answer_end, text):
        matches = []
        for m in re.finditer(re.escape(t), context, re.IGNORECASE):
            matches.append((m.start(), m.end()))

        # TODO Come back to later
        # if len(matches) > 1:
        #     print("Multiple matches found")

        # Only take match start closest to answer_start
        if len(matches) >= 1:
            abs_diff = [abs(m[0] - old_start) for m in matches]
            idx = abs_diff.index(min(abs_diff))
            new_start = matches[idx][0]
            new_end = matches[idx][1]
            updated_answer_start.append(new_start)
            updated_answer_end.append(new_end)
            updated_text.append(context[new_start:new_end])
        else:
            # Sometimes no matches are found (due to squad normalization)
            updated_answer_start.append(old_start)
            updated_answer_end.append(old_end)
            updated_text.append(t)

    return updated_text, updated_answer_start, updated_answer_end


def update_squad_normalization(
    context: str,
    text: List[str],
    answer_start: List[int],
    answer_end: List[int],
):
    # SQuAD normalization
    # 1. Check if text + context_ans match after normalization
    # 2. If yes then replace text with context_ans
    # 3. If no then we have normalization + wrong start
    updated_text = []
    for start, end, text in zip(answer_start, answer_end, text):
        context_ans = context[start:end]
        context_ans_1 = context[start:end+1]

        if normalize_answer(text) == normalize_answer(context_ans):
            updated_text.append(context_ans)
        elif normalize_answer(text) == normalize_answer(context_ans_1):
            updated_text.append(context_ans_1)
        else:
            updated_text.append(text)

    return updated_text


def update_answers_column(
    context: str,
    text: List[str],
    answer_start: List[int],
    answer_end: List[int],
):
    # Skip if all answers already found
    if all([t == context[start:start + len(t)] for start, t in zip(answer_start, text)]):
        return {"text": text, "answer_start": answer_start}

    # Answers differ due to squad normalization
    updated_text = update_squad_normalization(
        context=context, text=text, answer_start=answer_start, answer_end=answer_end
    )

    # Wrong start
    updated_text, updated_answer_start, updated_answer_end = update_answer_start(
        context=context, text=updated_text, answer_start=answer_start, answer_end=answer_end,
    )

    # Answers differ due to squad normalization
    updated_text = update_squad_normalization(
        context=context, text=updated_text, answer_start=updated_answer_start, answer_end=updated_answer_end
    )

    return {"text": updated_text, "answer_start": updated_answer_start}


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
    mrqa_datasets = mrqa_datasets.map(
        lambda example: {
            "answers": {
                "text": example["detected_answers"]["text"],
                "answer_start": [x["start"][0] for x in example["detected_answers"]["char_spans"]]
            }
        },
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
    ropes_datasets = ropes_datasets.map(
        lambda example: {"context": example["background"] + "\n\n " + example["situation"]},
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=["background", "situation"],
        load_from_cache_file=not overwrite_cache,
        desc="Creating context for ROPES",
    )

    def prepare_answers(examples):
        updated_examples = []
        for example in examples:
            context = example["context"]

            # Find all answer_starts
            answer_start = []
            for text in example["answers"]["text"]:
                answer_start.append(context.index(text))

            # Construct new answers column
            example["answers"] = {
                "text": example["answers"]["text"],
                "answer_start": []
            }
            updated_examples.append(example)
        return updated_examples

    # TODO Update answers to include answers_start and duplicates if answer appears more than once in the text.
    #      Will probably need a function instead of a lambda function
    ropes_datasets = ropes_datasets.map(
        prepare_answers,
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
