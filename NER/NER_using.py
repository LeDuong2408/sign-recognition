import os
import torch
from torch import nn
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerFast, DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from NER_dataset import load_ner_dataset
from NER_config import *
from NER_trainer import train_model, eval_model

from typing import List, Tuple

def ner_sentence(
    model: AutoModelForTokenClassification,
    tokenizer: PreTrainedTokenizerFast,
    sentence: str,
    device: torch.device = None
) -> List[Tuple[str, str]]:
    """
    Thực hiện NER inference trên 1 câu.
    Trả về list of (token, label).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 1. Tokenize
    encoding = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        is_split_into_words=False
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offsets = encoding["offset_mapping"][0].tolist()
    
    # 2. Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, num_labels]
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    
    # 3. Map back to tokens/words
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    results = []
    for token, pred_id, (start, end) in zip(tokens, pred_ids, offsets):
        # Skip special tokens
        if start == end:
            continue
        label = ID2LABELS[pred_id]
        results.append((token, label))
    
    return results


def ner_batch_sentence(
    model: AutoModelForTokenClassification,
    tokenizer: PreTrainedTokenizerFast,
    sentences: List[str],
    batch_size: int = 8,
    device: torch.device = None
) -> List[List[Tuple[str, str]]]:
    """
    Thực hiện NER inference trên 1 batch các câu.
    Trả về list-of-lists, mỗi phần tử là list of (token, label).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Data collator để padding dynamic
    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="pt")
    
    # Prepare dataset of tokenized inputs
    encodings = tokenizer(
        sentences,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
        is_split_into_words=False
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    offsets_batch = encodings["offset_mapping"]
     
    # Build DataLoader
    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, offsets_batch)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    all_results: List[List[Tuple[str, str]]] = []
    
    for batch in loader:
        batch_input_ids, batch_attention_mask, batch_offsets = [
            x.to(device) if x.dtype != torch.int64 else x for x in batch
        ]
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits  # [B, L, C]
            batch_pred = torch.argmax(logits, dim=-1).cpu().tolist()
        
        # Decode each sample in batch
        for preds, input_ids_row, offsets in zip(batch_pred,
                                                 batch_input_ids.cpu().tolist(),
                                                 batch_offsets.cpu().tolist()):
            tokens = tokenizer.convert_ids_to_tokens(input_ids_row)
            sample_res = []
            for token, pred_id, (start, end) in zip(tokens, preds, offsets):
                if start == end:
                    continue
                label = ID2LABELS[pred_id]
                sample_res.append((token, label))
            all_results.append(sample_res)
    
    return all_results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base-v2",
        num_labels=len(LABEL_LIST),
        id2label=ID2LABELS,
        label2id=LABELS2ID
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        PATH_SAVE_TOKENIZER, use_fast=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load("./NER/out/checkpoints/checkpoint_latest.pkl", map_location=torch.device(device))["model_state"])

    seq = "Tạp hóa Phố  Cổ  Địa chỉ: 77 Phố  Mã Mây, Phường Hàng Buồm, Quận Hoàn Kiếm, Hà Nội"
    result = ner_sentence(model, tokenizer, seq, device=device)
    name = ""
    for r in result:
        if "city" in r[1]:
            name = name + r[0]
    print(result)
    