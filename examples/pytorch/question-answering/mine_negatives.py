import random
import json

from datasets import load_dataset
from sentence_transformers import CrossEncoder


def find_random_negatives(qrels: dict, num_negatives: int = 5):
    """
    Add random negatives for each question in `qrels`. By default, 5 random negatives are added for each question.
    """
    qrels_with_negatives = qrels.copy()
    selected_questions = list(qrels.keys())
    for question in selected_questions:
        possible_keys = list(qrels.keys())
        # Remove documents that corresponds to the positive of that question.
        possible_keys.remove(question)

        sampled_keys = random.sample(possible_keys, num_negatives)
        negative_list = []
        negative_indices = []
        for key in sampled_keys:
            negative_list.append(qrels[key]["pos"][0])
            negative_indices.append(qrels[key]["pos_idx"][0])

        qrels_with_negatives[question]["neg"] = negative_list
        qrels_with_negatives[question]["neg_idx"] = negative_indices
    return qrels_with_negatives


def score_qrels(qrels: dict):
    """
    Score the qrels using a cross-encoder.
    """
    # qrels[question] = {"pos": [context], "pos_idx": [pos_idx], "neg": [neg_context], "neg_idx": [neg_idx]}
    qrels_with_scores = qrels.copy()

    # Prepare all tuples
    sentences = []
    idx_to_question = {}
    counter = 0
    for question in qrels:
        for pos in qrels[question]["pos"]:
            sentences.append([question, pos])
            idx_to_question[counter] = question
            counter += 1
        for neg in qrels[question]["neg"]:
            sentences.append([question, neg])
            idx_to_question[counter] = question
            counter += 1

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512)
    scores = model.predict(
        sentences=sentences,
        # sentences=[['Query', 'Paragraph1'], ['Query', 'Paragraph2'], ['Query', 'Paragraph3']],
        batch_size=32,
        num_workers=4,
    )
    # TODO Unpack the scores in their correct spots
    return qrels_with_scores


def filter_qrels(qrels: dict, margin: float = 3.):
    """
    Only keep negatives below a certain threshold with respect to the positive score.
    E.g. Threshold = 3, Pos score = 7, only keep as negatives of scores of 4 and below.
    """
    return qrels


def mine_negatives_adversarial_qa():
    # Notes and Qs:
    # - Only mine negatives from within the same dataset. Can then upload these as no answer questions as datasets for
    #   that specific dataset.
    # - Any point in mining negatives between datasets?

    # Let's make it easy on ourselves and just save in the final format of HF.
    # SynQA -> id, title, context, question, answers
    # Adversarial QA -> id, title, context, question, answers, metadata (only preserve split in metadata)
    # MRQA -> subset, context, context_tokens, qid, question, question_tokens, detected_answers, answers
    # 1. Load dataset for mining negatives
    adversarial_qa_datasets = load_dataset(
        "adversarial_qa",
        "adversarialQA",
        # cache_dir=cache_dir,
    )
    data = adversarial_qa_datasets["validation"]

    # 2. Construct qrels with positives
    qrels = {}
    for idx, row in enumerate(data):
        if row["question"] in qrels:
            qrels[row["question"]]["pos"].append(row["context"])
            qrels[row["question"]]["pos_idx"].append(idx)
        else:
            qrels[row["question"]] = {"pos": [row["context"]], "pos_idx": [idx], "neg": [], "neg_idx": []}

    # 3. Add random negatives to qrels
    qrels = find_random_negatives(qrels=qrels)

    # 4. Score the pairs
    qrels = score_qrels(qrels=qrels)

    # 5. Filter negatives
    qrels = filter_qrels(qrels=qrels)

    # 6. Save qrels in MS-Marco format (e.g. like hard negatives for MS-Marco)
    #    Will be an easier way to keep track of all cross-encoder scores for positive and negative examples.
    # with open("adversarial_qa_qrels.json") as f1:

    # 7. Save negatives jsonl format using HF column headers to be easily loadable for training
    #    Use id field to recreate all headers
    # Adversarial QA -> id, title, context, question, answers, metadata (only preserve split in metadata)
    counter = 0
    result = []
    for qrel in qrels:
        for negative_idx, negative in zip(qrel["neg_idx"], qrel["neg"]):
            title = data[negative_idx]["title"]
            row = {
                "id": counter,  # TODO Create unique hash
                "title": title,
                "context": negative,
                "answers": {"text": [], "answer_start": []},
                "metadata": {"split": ""}  # TODO Only retain {"split": "validation"} or {"split": "train"}
            }
            result.append(row)
            counter += 1

    with open("adversarial_qa_negatives.jsonl") as f1:
        for row in result:
            f1.write(json.dumps(row))
