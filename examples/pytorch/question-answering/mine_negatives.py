import random
import json

from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import CrossEncoder

from typing import Dict


def find_random_negatives(qrels: dict, num_negatives: int = 5):
    """
    Add random negative contexts for each question in `qrels`. By default, 5 random negative contexts are added for
    each question.
    """
    qrels_with_negatives = qrels.copy()
    # Mine 5 negative contexts per question in the dataset
    for question in qrels:
        # Remove contexts that corresponds to the positive of that question.
        possible_keys = list(qrels.keys())
        possible_keys.remove(question)

        sampled_keys = random.sample(possible_keys, num_negatives)
        negatives = []
        for key in sampled_keys:
            negatives.append(
                {
                    "question_id": qrels[question]["pos"][0]["question_id"],
                    "context_id": qrels[key]["pos"][0]["context_id"],
                    "context": qrels[key]["pos"][0]["context"]
                }
            )
        qrels_with_negatives[question]["neg"] = {"random": negatives}

    # qrels[question] = {
    #   "pos": [{"question_id": question_id, "context_id": context_id, "context": context}, ...],
    #   "neg": {
    #       "random": [{"question_id": question_id, "context_id": context_id, "context": context}, ...],
    #       ...
    #   },
    # }
    return qrels_with_negatives


def score_qrels(qrels: dict) -> Dict:
    """
    Score the qrels using a cross-encoder.
    """
    # qrels[question] = {
    #   "pos": [{"question_id": question_id, "context_id": context_id, "context": context}, ...],
    #   "neg": {
    #       "random": [{"question_id": question_id, "context_id": context_id, "context": context}, ...],
    #       ...
    #   },
    # }
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

    # qrels[question] = {
    #   "pos": [{"question_id": question_id, "context_id": context_id, "context": context, "ce-score": score}, ...],
    #   "neg": {
    #       "random": [{"question_id": question_id, "context_id": context_id, "context": context, "ce-score": score}, ...],
    #       ...
    #   },
    # }
    return qrels_with_scores


def get_hard_negatives(qrels: Dict, ce_score_margin: float = 3.):
    """
    Only keep negatives below a certain threshold with respect to the positive score.
    E.g. Threshold = 3, Pos score = 7, only keep as negatives of scores of 4 and below.

    Each positive and negative passage comes with a score from a Cross-Encoder (msmarco-MiniLM-L-6-v3). This allows denoising, i.e. removing false negative
    passages that are actually relevant for the query.
    """
    # qrels[question] = {
    #   "pos": [{"question_id": question_id, "context_id": context_id, "context": context, "ce-score": score}, ...],
    #   "neg": {
    #       "random": [{"question_id": question_id, "context_id": context_id, "context": context, "ce-score": score}, ...],
    #       ...
    #   },
    # }
    filtered_qrels = []
    for question, relations in tqdm(qrels.items()):
        # Get the positive passage ids
        pos_pids = [item['pid'] for item in relations['pos']]
        pos_scores = dict(zip(pos_pids, [item['ce-score'] for item in relations['pos']]))

        # Scoring
        pos_min_ce_score = min([item['ce-score'] for item in relations['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin

        # Get all the hard negatives
        neg_pids = set()
        neg_scores = {}
        for system_negs in relations['neg'].values():
            for item in system_negs:

                # Remove false negatives based on ce_score_threshold
                if item['ce-score'] > ce_score_threshold:
                    continue

                pid = item['pid']
                score = item['ce-score']
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    neg_scores[pid] = score

        if len(pos_pids) > 0 and len(neg_pids) > 0:
            filtered_qrels.append(
                {
                    'query_id': relations['qid'],
                    'pos': pos_pids,
                    'pos_scores': pos_scores,
                    'neg': list(neg_pids),
                    'neg_scores': neg_scores
                }
            )

    print(f"Train queries: {len(filtered_qrels)}")
    return filtered_qrels


# MRQA -> subset, context, context_tokens, qid, question, question_tokens, detected_answers, answers
# SynQA -> id, title, context, question, answers

def mine_negatives_adversarial_qa():
    # Notes and Qs:
    # - Only mine negatives from within the same dataset. Can then upload these as no answer questions as datasets for
    #   that specific dataset.
    # - Any point in mining negatives between datasets?
    # - Let's make it easy on ourselves and just save in the final format of the dataset as found on HuggingFace

    # 1. Load dataset for mining negatives
    split = "validation"
    adversarial_qa_datasets = load_dataset(
        "adversarial_qa",
        "adversarialQA",
        # cache_dir=cache_dir,
    )
    data = adversarial_qa_datasets[split]

    # 2. Construct qrels with positives
    # Adversarial QA only has one answer per row
    qrels = {}
    for row in data:
        if row["question"] in qrels:
            qrels[row["question"]]["pos"].append(
                {"question_id": row["id"], "context_id": row["id"], "context": row["context"]}
            )
        else:
            qrels[row["question"]] = {
                "pos": [{"question_id": row["id"], "context_id": row["id"], "context": row["context"]}],
                "neg": [],
            }

    # 3. Add random negatives to qrels
    qrels = find_random_negatives(qrels=qrels)

    # 4. Score the pairs
    qrels = score_qrels(qrels=qrels)

    # 5. Save qrels in MS-Marco format (e.g. like hard negatives for MS-Marco)
    #    Will be an easier way to keep track of all cross-encoder scores for positive and negative examples.
    # with open("adversarial_qa_qrels.json") as f1:

    # 6. Get hard negatives based on ce_score_margin
    #    This also combines and flattens all systems together
    qrels = get_hard_negatives(qrels=qrels, ce_score_margin=3.)

    # 7. Save negatives jsonl format using HF column headers to be easily loadable for training
    #    Use id field to recreate all headers
    # Adversarial QA -> id, title, context, question, answers, metadata
    # - title and context should be kept together
    counter = 0
    result = []
    for qrel in qrels:
        for negative in qrel["neg"]:
            title = data[negative["context_id"]]["title"]  # title and context are linked together
            row = {
                "id": counter,  # TODO Create new hash based on question_id and context_id
                "title": title,
                "context": negative["context"],
                "question": negative["question"],
                "answers": {"text": [], "answer_start": []},
                "metadata": {"split": split}  # only preserve split in metadata
            }
            result.append(row)
            counter += 1

    with open("adversarial_qa_negatives.jsonl") as f1:
        for row in result:
            f1.write(json.dumps(row))
