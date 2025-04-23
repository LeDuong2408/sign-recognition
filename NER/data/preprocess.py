from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
import json
import os
import random

INPUT_JSON = "./NER/data/1001.json"
OUTPUT_CONLL = "./NER/data/train1001.txt"

def create_json_file(path_raw_data, path_save_data):
    '''
    Create json file from raw data, this function using for label task by label studio tool
    :param path_raw_data: path to raw data file
    :param path_save_data: path to save json file
    '''
    with open(path_raw_data, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sentences = []
    for line in lines:
        line = line.strip()
        sentences.append(line)
    data = [{"text": s} for s in sentences]
    with open(path_save_data, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_train_data_conll(path_labeling_data, path_save_data):

    os.makedirs(os.path.dirname(path_save_data), exist_ok=True)

    # Load tokenizer PhoBERT
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "vinai/phobert-base", use_fast=True
    )

    with open(path_labeling_data, encoding="utf-8") as f, open(path_save_data, "w", encoding="utf-8") as out:
        tasks = json.load(f)
        for task in tasks:
            text = task["data"]["text"]

            char_labels = ["O"] * len(text)

            for ann in task["annotations"][0]["result"]:
                start, end = ann["value"]["start"], ann["value"]["end"]
                base_label = ann["value"]["labels"][0]  
               
                char_labels[start] = "B-" + base_label
                for i in range(start+1, end):
                    char_labels[i] = "I-" + base_label

            tok = tokenizer(text,
                            return_offsets_mapping=True,
                            padding=False,
                            truncation=False,
                            max_length=512)
            offsets = tok["offset_mapping"]
            tokens  = tok.tokens()

            # 4) Gán nhãn token bằng char_labels[offset_start]
            token_labels = []
            for (start, end), token in zip(offsets, tokens):
                if start is None or start >= len(char_labels):
                    token_labels.append("O")
                else:
                    token_labels.append(char_labels[start])

            # 5) Xuất CoNLL
            for token, label in zip(tokens, token_labels):
                out.write(f"{token} {label}\n")
            out.write("\n")

    print(f"Processed and saved to {path_save_data}")

def split_train_val_data(path_data, path_train, path_val, val_size=0.1):
    with open(path_data, "r", encoding="utf-8") as f:
        lines = f.readlines()
    num_val = int(len(lines) * val_size)
    with open(path_train, "w", encoding="utf-8") as f:
        f.writelines(lines[:-num_val])
    with open(path_val, "w", encoding="utf-8") as f:
        f.writelines(lines[-num_val:])

if __name__ == "__main__":
    # create_json_file("./NER/data/text.txt", "./NER/data/hoang.json")
    # create_train_data_conll(INPUT_JSON, OUTPUT_CONLL)
    split_train_val_data(   path_data = "./NER/data/train1001.txt", 
                            path_train = "./NER/data/train.txt",
                            path_val = "./NER/data/val.txt",
                            val_size = 0.1)