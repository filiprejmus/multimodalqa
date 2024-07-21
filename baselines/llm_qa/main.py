from model import OpenAIInferer
from dataset import DataSetManager
from common_utils import read_jsonl
import os
from sklearn.metrics import f1_score
import asyncio

DATA_DIR = "downloads/MMQA"


def compute_exact_match(infered_answers, dev_answers):
    infered = sorted([answer.lower() for answer in infered_answers.get("answers", [])])
    dev = sorted([answer.lower() for answer in dev_answers.get("answers", [])])
    return infered == dev


def compute_f1_score(infered_answers, dev_answers):
    infered = [answer.lower() for answer in infered_answers.get("answers", [])]
    dev = [answer.lower() for answer in dev_answers.get("answers", [])]

    # Flatten the lists to compute F1 score
    infered_flat = [item for sublist in infered for item in sublist]
    dev_flat = [item for sublist in dev for item in sublist]

    # Compute F1 score for multiple answers
    infered_set = set(infered_flat)
    dev_set = set(dev_flat)

    true_positives = len(infered_set & dev_set)
    false_positives = len(infered_set - dev_set)
    false_negatives = len(dev_set - infered_set)

    if true_positives == 0:
        return 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return 2 * (precision * recall) / (precision + recall)


def processAnswers(answers):
    answers_processed = []
    for a in answers:
        answers_processed.append(a["answer"])
    return {"answers": answers_processed}


if __name__ == "__main__":
    inferer = OpenAIInferer()
    dataset = DataSetManager(DATA_DIR)

    print("Current working directory:", os.getcwd())
    data = read_jsonl("downloads/MMQA/MMQA_dev.jsonl")

    inferences = []
    expected_answers = {}
    i = 0
    for d in data:
        i += 1
        textids = d["metadata"]["text_doc_ids"]
        imageids = d["metadata"]["image_doc_ids"]
        tableid = d["metadata"]["table_id"]
        question = d["question"]
        answers = d["answers"]
        qid = d["qid"]

        tb_ctx = dataset.createTableContext(tableid)
        img_ctx = dataset.createImageContext(imageids)
        txt_ctx = dataset.createTextContext(textids)

        inferences.append((txt_ctx, img_ctx, tb_ctx, question, qid))
        expected_answers[qid] = answers
        if i == 3:
            break

    results = asyncio.run(inferer.run_inferences(inferences))
    for result in results:
        qid = result["qid"]
        infered_answers = result["result"]
        price_in_usd = result["price_in_usd"]

        dev_answers = processAnswers(expected_answers[qid])
        print("Expected answers: ", dev_answers["answers"])
        print("Infered answers: ", infered_answers["answers"])
        exact_match = compute_exact_match(infered_answers, dev_answers)
        f1 = compute_f1_score(infered_answers, dev_answers)
        print(f"Exact Match: {exact_match}")
        print(f"F1 Score: {f1}")
