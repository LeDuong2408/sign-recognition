import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        return item
    
def load_ner_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizerFast,
    label2id: dict,
    max_length: int = 128
):
    """
    Đọc dữ liệu CoNLL từ data_path, tokenize và align label.
    
    Args:
      - data_path: đường dẫn file .txt dạng CoNLL
      - tokenizer: tokenizer của phoBERT (Fast tokenizer)
      - label2id: dict mapping label->id
      - max_length: độ dài tối đa cho tokenizer.pad/truncate
    
    Returns:
      - dataset: torch Dataset với __getitem__ trả về dict {
            'input_ids','attention_mask','labels'
        }
    """
    sentences = []
    labels = []
    with open(data_path, encoding="utf-8") as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                splits = line.split()
                tokens.append(splits[0])
                tags.append(splits[-1])

        if tokens:
            sentences.append(tokens)
            labels.append(tags)
    
    encodings = tokenizer(
        sentences,
        is_split_into_words=True,
        return_attention_mask=True, # can compare token padding and true token
        padding=False,
        truncation=True,
        max_length=max_length
    )

    all_label_ids = []
    for i, tag_seq in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(label2id[tag_seq[word_idx]])
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx
        all_label_ids.append(label_ids)
    encodings["labels"] = all_label_ids

    # Dataset
    return NERDataset(encodings)
