import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

def load_ner_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizerFast,
    label2id: dict,
    max_length: int = 128
) -> NERDataset:
    """
    Đọc file CoNLL từ data_path, tokenize và align label cho ner-vietnamese-electra-base.

    Args:
      - data_path: đường dẫn file .conll / .txt dạng CoNLL (token label trên mỗi dòng, cách câu bằng dòng trống)
      - tokenizer: tokenizer Fast (ví dụ AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base"))
      - label2id: dict mapping label->id (vd: {"O":0, "B-LOCATION":1, ...})
      - max_length: độ dài tối đa cho tokenizer padding/truncate

    Returns:
      - NERDataset với các trường input_ids, attention_mask, labels (ignore_index=-100)
    """
    sentences, tag_seqs = [], []
    tokens, tags = [], []

    # 1) Đọc file CoNLL, gom thành từng câu
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # mỗi dòng: "token label"
                parts = line.split()
                tokens.append(parts[0])
                tags.append(parts[-1])
            else:
                # kết thúc 1 câu
                if tokens:
                    sentences.append(tokens)
                    tag_seqs.append(tags)
                    tokens, tags = [], []
        # xử lý câu cuối nếu file không kết thúc bằng dòng trống
        if tokens:
            sentences.append(tokens)
            tag_seqs.append(tags)

    # 2) Tokenize & lấy word_ids để align
    encodings = tokenizer(
        sentences,
        is_split_into_words=True,
        return_attention_mask=True,
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    # 3) Align nhãn từ tag_seqs sang encodings["input_ids"]
    all_label_ids = []
    for i, tags in enumerate(tag_seqs):
        word_ids = encodings.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # token padding hoặc special token
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                # token đầu của một từ
                label_ids.append(label2id[tags[word_idx]])
            else:
                # các sub-token sau, ignore
                label_ids.append(-100)
            prev_word_idx = word_idx
        all_label_ids.append(label_ids)

    encodings["labels"] = all_label_ids

    return NERDataset(encodings)
