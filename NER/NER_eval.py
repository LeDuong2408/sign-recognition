import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForTokenClassification, pipeline
import re
import glob
import os
import torch
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")

def extract_entities(text):
    entity_spans = []
    clean_text_parts = []
    cursor = 0

    # Iterate over all ENAMEX tags
    for match in re.finditer(r'<ENAMEX\s+TYPE="(.*?)">(.*?)</ENAMEX>', text, flags=re.DOTALL):
        ent_type, ent_text = match.group(1), match.group(2)
        # Append text before the tag
        clean_text_parts.append(text[cursor:match.start()])
        # Record offsets in the cleaned text
        start_offset = sum(len(part) for part in clean_text_parts)
        clean_text_parts.append(ent_text)
        end_offset = sum(len(part) for part in clean_text_parts)
        entity_spans.append((start_offset, end_offset, ent_type))
        cursor = match.end()

    # Append any remaining text after the last tag
    clean_text_parts.append(text[cursor:])
    clean_text = ''.join(clean_text_parts)
    return clean_text, entity_spans 

def conll_format(text, entity_spans):
    all_tokens = []
    all_labels = []
    len_text = len(text)
    lin_space = np.arange(0, len_text + 1, 512, dtype=int)
    
    for i in range(len(lin_space) - 1):
        start = lin_space[i]
        end = lin_space[i + 1]
        text_part = text[start:end]

        encoding = tokenizer(text_part, return_offsets_mapping=True,
                            add_special_tokens=False,)
        
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        offsets = encoding['offset_mapping'] # offsets is start and end of tokens in text_part
        labels = ["O"] * len(tokens)

        for ent_start, ent_end, label in entity_spans:

            if ent_start >= start and ent_end <= end:
                relative_start = ent_start - start
                relative_end = ent_end - start
                for idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start >= relative_start and tok_end <= relative_end:
                        labels[idx] = f"B-{label}" if tok_start == relative_start else f"I-{label}"

        all_labels.extend(labels)
        all_tokens.extend(tokens)
    return all_tokens, all_labels

def predict_align_labels(model, text):
    alligned_labels = []
    len_text = len(text)
    lin_space = np.arange(0, len_text + 1, 512, dtype=int)
    for i in range(len(lin_space) - 1):
        start = lin_space[i]
        end = lin_space[i + 1]
        text_part = text[start:end]
        output = model(text_part)
        
        encoding = tokenizer(text_part, return_offsets_mapping=True,
                        add_special_tokens=False,)
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        num_tokens = len(tokens)
        part_label = ["O"] * num_tokens

        for out in output:
            part_label[out['index']-1] = out['entity']
        alligned_labels.extend(part_label)

    return alligned_labels

def predict_align_labels2(model, text):
    alligned_labels = []
    output = model(text)
    
    encoding = tokenizer(text, return_offsets_mapping=True,
                    add_special_tokens=False,)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    num_tokens = len(tokens)
    part_label = ["O"] * num_tokens

    for out in output:
        part_label[out['index']-1] = out['entity']
    alligned_labels.extend(part_label)
    return alligned_labels

def read_all_data(path_data, model, model_finetuned):
    all_tokens = []
    all_true_labels = []
    all_pred_labels = []
    all_pred_labels_finetuned = []
    categories = glob.glob(path_data + "/**/*.txt", recursive=True)
    for file in tqdm(categories, desc="Processing files"):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            clean_text, entity_spans = extract_entities(text)
            tokens, true_labels = conll_format(clean_text, entity_spans)
            pred_labels = predict_align_labels(model, clean_text)
            pred_labels_finetuned = predict_align_labels(model_finetuned, clean_text)
            assert len(tokens) == len(true_labels) == len(pred_labels) == len(pred_labels_finetuned), "Length mismatch"
            all_tokens.extend(tokens)
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
            all_pred_labels_finetuned.extend(pred_labels_finetuned)

    return all_tokens, all_true_labels, all_pred_labels, all_pred_labels_finetuned

def GenTokenNSavePred_VLSP2018(path_data, model, model_finetuned):
    tokens, true_labels, pred_labels, pred_labels_finetuned = read_all_data(path_data, model, model_finetuned)
    df = pd.DataFrame({
        "tokens": tokens,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "pred_labels_finetuned": pred_labels_finetuned,
    })

    # Đảm bảo folder đầu ra tồn tại
    out_dir = "./NER/data/"
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, "result_VLSP2018.csv")
    df.to_csv(out_file, index=False, encoding="utf-8")
    return df

def compute_metrics(true_labels,pred_labels):

    if isinstance(pred_labels[0], str):
        pred_labels = [pred_labels]
        true_labels = [true_labels]

    # return {
    #     "precision": precision_score(true_labels, pred_labels),
    #     "recall": recall_score(true_labels, pred_labels),
    #     "f1": f1_score(true_labels, pred_labels)
    # }
    return classification_report(true_labels, pred_labels)

def GenTokenNSavePred(path_data, model, model_finetuned):
    all_tokens = []
    all_true_labels = []
    all_pred_labels = []
    all_pred_labels_finetuned = []
    with open(path_data, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        clean_text = ""
        tokens = []
        true_labels = []
        pred_labels = []
        pred_labels_finetuned = []

        for line in tqdm(all_lines, desc="Processing lines"):
            if line.strip():
                data = line.strip().split()
                tokens.append(data[0])
                true_labels.append(data[1])
                # if data[0] != "[CLS]" and data[0] != "[SEP]":
                if "##" in data[0]:
                    clean_text += data[0].replace("##", "") 
                else:
                    clean_text += " " + data[0]
            else:
                try: 
                    if clean_text:
                        pred_labels = predict_align_labels2(model, clean_text)
                        pred_labels_finetuned = predict_align_labels2(model_finetuned, clean_text)
                        assert len(tokens) == len(true_labels) == len(pred_labels) == len(pred_labels_finetuned), f"Length mismatch{len(tokens), len(true_labels), len(pred_labels), len(pred_labels_finetuned)}"
                        all_tokens.extend(tokens)
                        all_true_labels.extend(true_labels)                    
                        all_pred_labels.extend(pred_labels)
                        all_pred_labels_finetuned.extend(pred_labels_finetuned)
                        clean_text = ""
                        tokens = []
                        true_labels = []
                except:
                    print(f"Length mismatch{len(tokens), len(true_labels), len(pred_labels), len(pred_labels_finetuned)}")
                    continue
    
    df = pd.DataFrame({
        "tokens": all_tokens,
        "true_labels": all_true_labels,
        "pred_labels": all_pred_labels,
        "pred_labels_finetuned": all_pred_labels_finetuned,
    })

    # Đảm bảo folder đầu ra tồn tại
    out_dir = "./NER/data/"
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, "result.csv")
    df.to_csv(out_file, index=False, encoding="utf-8")
    return df
                
if __name__ == "__main__":
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    model_finetuned = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    
    # Load the fine-tuned model
    checkpoint = torch.load("./NER/out/checkpoints/checkpoint_best.pkl", map_location="cpu")
    state_dict = checkpoint["model_state"]
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v
    model_finetuned.load_state_dict(new_state)

    # Create pipelines for both models
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    nlp_finetuned = pipeline("ner", model=model_finetuned, tokenizer=tokenizer)

    # Evaluate the models
    path_eval = "./NER/data/electra-vn/eval_electra.txt"
    result = GenTokenNSavePred(path_data=path_eval, model=nlp, model_finetuned=nlp_finetuned) 
    
    true_labels = result["true_labels"].tolist()
    pred_labels = result["pred_labels"].tolist()
    pred_labels_finetuned = result["pred_labels_finetuned"].tolist()

    metrics = compute_metrics(true_labels, pred_labels)
    metrics_finetuned = compute_metrics(true_labels, pred_labels_finetuned)
    metrics_all = f"Metrics for original model:\n {metrics}"\
                + "\nMetrics for finetuned model:\n" + metrics_finetuned
    with open("./NER/data/metrics.txt", "w", encoding="utf-8") as f:
        f.write(metrics_all)


    # path_data = "./NER/data/VLSP2018_Test"
    # result = GenTokenNSavePred_VLSP2018(path_data=path_data, model=nlp, model_finetuned=nlp_finetuned)    
    
    result = pd.read_csv("./NER/data/result_VLSP2018.csv", encoding="utf-8")
    true_labels = result["true_labels"].tolist()
    pred_labels = result["pred_labels"].tolist()
    pred_labels_finetuned = result["pred_labels_finetuned"].tolist()

    metrics = compute_metrics(true_labels, pred_labels)
    metrics_finetuned = compute_metrics(true_labels, pred_labels_finetuned)
    metrics_all = f"Metrics for original model:\n {metrics}"\
                + "\nMetrics for finetuned model:\n" + metrics_finetuned
    with open("./NER/data/metrics_VLSP2018.txt", "w", encoding="utf-8") as f:
        f.write(metrics_all)

