from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
import json
import os
import random
from typing import List

INPUT_JSON = "./NER/data/hoang536.json"
OUTPUT_CONLL = "./NER/data/hoang536.txt"

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
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(
    #     "vinai/phobert-base", use_fast=True
    # )
    tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    merge_map = {
        "city":       "LOCATION",
        "ward":       "LOCATION",
        "district":   "LOCATION",
        "village":    "LOCATION",
        "street":     "LOCATION",
        "province":   "LOCATION",
        "country":    "LOCATION",
        "postcode":   "LOCATION",
        "nameshop":   "ORGANIZATION",
        "store":      "ORGANIZATION",
        "company":    "ORGANIZATION",
        "brand":      "ORGANIZATION",
    }

    with open(path_labeling_data, encoding="utf-8") as f_in, \
         open(path_save_data, "w", encoding="utf-8") as f_out:

        tasks = json.load(f_in)
        for task in tasks:
            text = task["data"]["text"]
            length = len(text)

            char_labels = ["O"] * length

            for ann in task["annotations"][0]["result"]:
                s, e = ann["value"]["start"], ann["value"]["end"]
                orig = ann["value"]["labels"][0]  # ví dụ "street", "nameshop",…
                mapped = merge_map.get(orig, orig.upper())
                # Gán B- và I-
                char_labels[s] = f"B-{mapped}"
                for i in range(s + 1, min(e, length)):
                    char_labels[i] = f"I-{mapped}"

            # Tokenize để lấy offsets
            encoding = tokenizer(
                text,
                return_offsets_mapping=True,
                padding=False,
                truncation=False,
                max_length=512
            )
            offsets = encoding["offset_mapping"]
            tokens  = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

            # Gán nhãn token-level dựa vào nhãn char_labels
            token_labels = []
            for (start, end), token in zip(offsets, tokens):
                if start is None or start >= length:
                    token_labels.append("O")
                else:
                    token_labels.append(char_labels[start])

            # 5) Xuất CoNLL: mỗi dòng "token label", cách đoạn bằng dòng trắng
            for token, label in zip(tokens, token_labels):
                f_out.write(f"{token} {label}\n")
            f_out.write("\n")

    print(f"Processed and saved to {path_save_data}")

def split_train_val_data(
    path_data: str,
    path_train: str,
    path_val: str,
    val_size: float = 0.1,
    seed: int = 42
):
    """
    Đọc file CoNLL ở path_data, tách thành block (câu),
    chia ngẫu nhiên thành train và val, rồi ghi ra path_train & path_val.

    Args:
      - path_data: đường dẫn file CoNLL đầy đủ
      - path_train: file sẽ ghi train
      - path_val:   file sẽ ghi val
      - val_size:   tỉ lệ block dùng cho validation (ví dụ 0.1 = 10%)
      - seed:       seed cho random.shuffle để tái lập kết quả
    """
    with open(path_data, "r", encoding="utf-8") as f:
        blocks: List[List[str]] = []
        current: List[str] = []
        for line in f:
            if line.strip():
                current.append(line)
            else:
                if current:
                    blocks.append(current.copy())
                    current.clear()

        if current:
            blocks.append(current.copy())

    random.seed(seed)
    random.shuffle(blocks)

    num_blocks = len(blocks)
    num_val = int(num_blocks * val_size)
    val_blocks = blocks[:num_val]
    train_blocks = blocks[num_val:]

    os.makedirs(os.path.dirname(path_train), exist_ok=True)
    os.makedirs(os.path.dirname(path_val), exist_ok=True)

    def write_blocks(path: str, blk_list: List[List[str]]):
        with open(path, "w", encoding="utf-8") as fw:
            for blk in blk_list:
                for l in blk:
                    fw.write(l)
                fw.write("\n")  

    write_blocks(path_train, train_blocks)
    write_blocks(path_val,   val_blocks)

    print(f"Đã chia {num_blocks} block thành {len(train_blocks)} train / {len(val_blocks)} val.")

def clear_punctuation(path_data):
    all_data = []
    with open(path_data, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                if line[0] == "," or line[0] == "." or line[0] == "!" or line[0] == "?" or line[0] == ":":
                    continue
            all_data.append(line)

    with open(path_data, "a", encoding="utf-8") as f:
        for line in all_data:
            f.write(line + "\n")

if __name__ == "__main__":
    # create_json_file("./NER/data/text.txt", "./NER/data/hoang.json")
    # create_train_data_conll(INPUT_JSON, OUTPUT_CONLL)
    # clear_punctuation("./NER/data/electra-vn/all_electra.txt")
    split_train_val_data(   path_data = "./NER/data/electra-vn/all_electra.txt", 
                            path_train = "./NER/data/electra-vn/train_electra.txt",
                            path_val = "./NER/data/electra-vn/eval_electra.txt",
                            val_size = 0.1)