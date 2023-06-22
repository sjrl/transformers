import re
import string
from typing import Optional, List

from datasets import load_dataset, concatenate_datasets, DatasetDict, Sequence, Value


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
    updated_answer_start: List[int] = []
    updated_answer_end: List[int] = []
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
            new_start = int(matches[idx][0])
            new_end = int(matches[idx][1])
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
    """
    SQuAD normalization
    1. Check if text + answer from the context match after normalization
    2. If yes then replace text with the answer from the context
    3. If no then we can still have normalization + wrong start
    """
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

    # Answers can differ due to squad normalization
    updated_text = update_squad_normalization(
        context=context, text=text, answer_start=answer_start, answer_end=answer_end
    )

    # Some answers have the wrong start
    updated_text, updated_answer_start, updated_answer_end = update_answer_start(
        context=context, text=updated_text, answer_start=answer_start, answer_end=answer_end,
    )

    # TODO Check if this is really necessary
    # After updating answer start recheck if answers differ due to squad normalization
    updated_text = update_squad_normalization(
        context=context, text=updated_text, answer_start=updated_answer_start, answer_end=updated_answer_end
    )

    return {"text": updated_text, "answer_start": updated_answer_start}


def check_answer_in_context(
    context: str,
    text: List[str],
    answer_start: List[int],
    idx: int,
    subset: str,
):
    """
    Checks if the answer (text) is exactly matches the one found in the context.
    """
    not_found_text = []
    not_found_context_answers = []
    for start, t in zip(answer_start, text):
        context_ans = context[start:start + len(t)]
        if t != context_ans:
            not_found_context_answers.append(context_ans)
            not_found_text.append(t)

    if len(not_found_text) > 0:
        for t, c in zip(not_found_text, not_found_context_answers):
            print(f"Idx: {idx} Subset: {subset} Answer: '{t}' vs. '{c}'")


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


# TODO Consider handling the case of multiple matches.
#      Start with looking at some examples to see what we are working with.
# TODO Remove special tokens [DOC], [TLE] and [PAR]. They seem to be inconsistently applied
def _prep_mrqa(
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
    preprocessing_num_workers: int = 1,
    overwrite_cache: bool = False,
):
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

    def update_answers(example, idx):
        text = example["detected_answers"]["text"]
        answer_start = [int(x["start"][0]) for x in example["detected_answers"]["char_spans"]]
        answer_end = [int(x["end"][0]) for x in example["detected_answers"]["char_spans"]]

        assert len(text) == len(answer_start), "Check that text and answer_start have same length"
        text_and_start = update_answers_column(
            context=example["context"],
            text=text,
            answer_start=answer_start,
            answer_end=answer_end,
        )
        text = text_and_start["text"]
        answer_start = text_and_start["answer_start"]

        check_answer_in_context(
            context=example["context"],
            text=text,
            answer_start=answer_start,
            idx=idx,
            subset=example["subset"]
        )

        example["answers"] = {"text": text, "answer_start": answer_start}
        return example

    # Map detected_answers to answers in the correct format
    mrqa_datasets = mrqa_datasets.map(
        update_answers,
        with_indices=True,
        num_proc=preprocessing_num_workers,
        remove_columns=["detected_answers"],
        load_from_cache_file=not overwrite_cache,
        desc="Formatting MRQA",
    )
    mrqa_datasets = mrqa_datasets.cast_column(
        "answers",
        Sequence(
            feature={
                'text': Value(dtype='string', id=None),
                'answer_start': Value(dtype='int32', id=None)
            },
            length=-1, id=None
        )
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

    # Create context string
    ropes_datasets = ropes_datasets.map(
        lambda example: {"context": example["background"] + "\n\n " + example["situation"]},
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=["background", "situation"],
        load_from_cache_file=not overwrite_cache,
        desc="Creating context for ROPES",
    )

    def prepare_answers(example):
        # TODO Update answers to include answers_start and duplicates if answer appears more than once in the text.
        # Find all answer_starts
        answer_start = []
        context = example["context"]
        for text in example["answers"]["text"]:
            answer_start.append(context.index(text))

        # Construct new answers column
        example["answers"] = {
            "text": example["answers"]["text"],
            "answer_start": answer_start
        }
        return example

    ropes_datasets = ropes_datasets.map(
        prepare_answers,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Updating answers for ROPES",
    )
    ropes_datasets = _add_subset_column(dataset_dict=ropes_datasets, subset_name="ROPES")

    def check_answers(example, idx):
        text = example["answers"]["text"]
        answer_start = example["answers"]["answer_start"]
        assert len(text) == len(answer_start), "Check that text and answer_start have same length"
        check_answer_in_context(
            context=example["context"],
            text=text,
            answer_start=answer_start,
            idx=idx,
            subset=example["subset"]
        )
        return example
    ropes_datasets = ropes_datasets.map(
        check_answers,
        with_indices=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Checking ROPES",
    )

    return ropes_datasets


def _prep_squad_v2(
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
    preprocessing_num_workers: int = 1,
    overwrite_cache: bool = False,
):
    # Has train, validation
    squad_v2_datasets = load_dataset(
        "squad_v2",
        None,
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    squad_v2_datasets = _add_subset_column(dataset_dict=squad_v2_datasets, subset_name="SQuADV2")
    squad_v2_datasets = squad_v2_datasets.remove_columns("title")

    def check_answers(example, idx):
        text = example["answers"]["text"]
        answer_start = example["answers"]["answer_start"]
        assert len(text) == len(answer_start), "Check that text and answer_start have same length"
        check_answer_in_context(
            context=example["context"],
            text=text,
            answer_start=answer_start,
            idx=idx,
            subset=example["subset"]
        )
        return example
    squad_v2_datasets = squad_v2_datasets.map(
        check_answers,
        with_indices=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Checking SQuADV2",
    )

    return squad_v2_datasets


def _prep_adversarial_qa(
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
    preprocessing_num_workers: int = 1,
    overwrite_cache: bool = False,
):
    # Has train, validation, test
    adversarial_qa_datasets = load_dataset(
        "adversarial_qa",
        "adversarialQA",
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    # Remove extra column
    adversarial_qa_datasets = adversarial_qa_datasets.remove_columns("metadata")
    adversarial_qa_datasets = adversarial_qa_datasets.remove_columns("title")
    adversarial_qa_datasets = _add_subset_column(dataset_dict=adversarial_qa_datasets, subset_name="adversarialQA")

    def check_answers(example, idx):
        text = example["answers"]["text"]
        answer_start = example["answers"]["answer_start"]
        assert len(text) == len(answer_start), "Check that text and answer_start have same length"
        check_answer_in_context(
            context=example["context"],
            text=text,
            answer_start=answer_start,
            idx=idx,
            subset=example["subset"]
        )
        return example
    adversarial_qa_datasets = adversarial_qa_datasets.map(
        check_answers,
        with_indices=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Checking adversarialQA",
    )

    return adversarial_qa_datasets


def _prep_synqa(
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
    preprocessing_num_workers: int = 1,
    overwrite_cache: bool = False,
):
    synqa_datasets = load_dataset(
        "mbartolo/synQA",
        None,
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    synqa_datasets = _add_subset_column(dataset_dict=synqa_datasets, subset_name="synQA")
    synqa_datasets = synqa_datasets.remove_columns("title")

    def check_answers(example, idx):
        text = example["answers"]["text"]
        answer_start = example["answers"]["answer_start"]
        assert len(text) == len(answer_start), "Check that text and answer_start have same length"

        check_answer_in_context(
            context=example["context"],
            text=text,
            answer_start=answer_start,
            idx=idx,
            subset=example["subset"]
        )

        return example

    # Map detected_answers to answers in the correct format
    synqa_datasets = synqa_datasets.map(
        check_answers,
        with_indices=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Checking synQA",
    )

    return synqa_datasets


# TODO Check average context length to know how big context length for the model should be.
#      According to MRQA page, can expect context lengths of up to 800 tokens.
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
    # ropes_datasets = _prep_ropes(
    #     cache_dir=cache_dir,
    #     use_auth_token=use_auth_token,
    #     preprocessing_num_workers=preprocessing_num_workers,
    #     overwrite_cache=overwrite_cache
    # )

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
            squad_v2_datasets["train"],
            adversarial_qa_datasets["train"],
            synqa_datasets["train"],
            # ropes_datasets["train"],
        ]
    )
    validation = concatenate_datasets(
        [
            mrqa_datasets["validation"],
            squad_v2_datasets["validation"],
            adversarial_qa_datasets["validation"],
            # ropes_datasets["validation"],
            # synqa_datasets["validation"]  # Doesn't exist
        ]
    )
    test = concatenate_datasets(
        [
            mrqa_datasets["test"],
            adversarial_qa_datasets["test"],
            # ropes_datasets["test"],
            # squad_v2_datasets["test"],  # N/A
            # synqa_datasets["test"]  # Doesn't exist
        ]
    )
    return DatasetDict({"train": train, "validation": validation, "test": test})
