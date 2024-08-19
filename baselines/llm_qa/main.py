from model import OpenAIInferer
from dataset import DataSetManager
from common_utils import read_jsonl
import os
from sklearn.metrics import f1_score
import asyncio
import logging
import json

logging.basicConfig(
    filename="inference_results.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)


DATA_DIR = "downloads/MMQA"
OUTPUT_FILE = "inference_results.json"


def compute_exact_match(infered_answers, dev_answers):
    infered = sorted(
        [str(answer).lower() for answer in infered_answers.get("answers", [])]
    )
    dev = sorted([str(answer).lower() for answer in dev_answers.get("answers", [])])
    return infered == dev


def compute_f1_score(infered_answers, dev_answers):
    infered = [str(answer).lower() for answer in infered_answers.get("answers", [])]
    dev = [str(answer).lower() for answer in dev_answers.get("answers", [])]

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


async def process_batch(batch, inferer, dataset, expected_answers, batch_id):
    inferences = []
    results_list = []

    for d in batch:
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

    results = await inferer.run_inferences(inferences)
    for result in results:
        qid = result["qid"]
        infered_answers = result["result"]
        price_in_usd = result["price_in_usd"]

        dev_answers = processAnswers(expected_answers[qid])
        exact_match = compute_exact_match(infered_answers, dev_answers)
        f1 = compute_f1_score(infered_answers, dev_answers)
        result_entry = {
            "batch_id": batch_id,
            "qid": qid,
            "expected_answers": dev_answers["answers"],
            "infered_answers": infered_answers["answers"],
            "exact_match": exact_match,
            "f1_score": f1,
            "price_in_usd": price_in_usd,
        }
        results_list.append(result_entry)

    with open(OUTPUT_FILE, "a") as f:
        for result in results_list:
            f.write(json.dumps(result) + "\n")


async def runner(data, inferer, dataset):
    expected_answers = {}
    batch_size = 60
    delay = 65
    for i in range(2052, len(data), batch_size):
        batch_id = i // batch_size + 1
        batch = data[i : i + batch_size]
        await process_batch(batch, inferer, dataset, expected_answers, batch_id)
        if i + batch_size < len(data):
            print(f"Waiting for {delay} seconds before processing the next batch...")
            await asyncio.sleep(delay)


if __name__ == "__main__":
    inferer = OpenAIInferer()
    dataset = DataSetManager(DATA_DIR)

    print("Current working directory:", os.getcwd())
    data = read_jsonl("downloads/MMQA/MMQA_dev.jsonl")
    asyncio.run(runner(data, inferer, dataset))
