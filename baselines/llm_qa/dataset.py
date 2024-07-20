from typing import List
from common_utils import read_jsonl
from PIL import Image
import base64
import io
import os


class DataSetManager:

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.text_data = self.read_text_docs()
        self.image_data = self.read_image_infos()
        self.table_data = self.read_table_infos()

    def read_data(self, file_name: str):
        return read_jsonl(f"{self.data_dir}/{file_name}")

    def read_text_docs(self):
        text_data = self.read_data("MMQA_texts.jsonl")
        return {doc["id"]: doc for doc in text_data}

    def read_image_infos(self):
        image_data = self.read_data("MMQA_images.jsonl")
        return {info["id"]: info for info in image_data}

    def read_table_infos(self):
        table_data = self.read_data("MMQA_tables.jsonl")
        return {info["id"]: info for info in table_data}

    def createTextContext(self, textIDs: List[str]):
        texts = []
        for textID in textIDs:
            texts.append(
                self.text_data[textID]["title"]
                + "\n"
                + self.text_data[textID]["text"]
                + "\n"
            )
        return texts

    def createImageContext(self, imageIDs: List[str]):
        images = []
        for imageID in imageIDs:
            img_path = self.image_data[imageID]["path"]
            image = load_and_encode_image(
                os.path.join(self.data_dir, "final_dataset_images", img_path)
            )
            title = self.image_data[imageID]["title"]
            images.append({"title": title, "image": image})
        return images

    def createTableContext(self, tableID: str):
        table = self.table_data[tableID]
        header = table["table"]["header"]
        table_rows = table["table"]["table_rows"]
        texts = extract_text_from_nested(table_rows)
        return table["title"] + "\n" + str(header) + "\n" + str(texts)


def load_and_encode_image(image_path):
    # Load the image
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return img_base64


def extract_text_from_nested(data):
    if isinstance(data, dict):
        if "text" in data:
            return data["text"]
        else:
            return {key: extract_text_from_nested(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [extract_text_from_nested(item) for item in data]
    else:
        return data
