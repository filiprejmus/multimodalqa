from model import OpenAIInferer
from dataset import DataSetManager
from common_utils import read_jsonl
import os
import base64

DATA_DIR = "downloads/MMQA"

if __name__ == "__main__":
    inferer = OpenAIInferer()
    dataset = DataSetManager(DATA_DIR)

    print("Current working directory:", os.getcwd())
    data = read_jsonl("downloads/MMQA/MMQA_dev.jsonl")

    for d in data:
        textids = d["metadata"]["text_doc_ids"]
        imageids = d["metadata"]["image_doc_ids"]
        tableid = d["metadata"]["table_id"]
        break
    dataset.createTableContext(tableid)

# print(data[0])
