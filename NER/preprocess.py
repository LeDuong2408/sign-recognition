from transformers import AutoTokenizer

def prepare_data(path_raw_data, path_save_data):
    with open(path_raw_data, "r", encoding="utf-8") as f:
        lines = f.readlines()  # read all line into list

    for i in range(0, len(lines), 2):
        print(lines[i]) 
        if i + 1 < len(lines):  # Kiểm tra tránh lỗi
            print(lines[i + 1].strip())
        print(i)
        

# prepare_data("data/raw_data.txt", "data/train.json")


def tokenize_and_align_labels(tokenizer, sentences, labels, label2id, max_length=128):
    tokenized_inputs = tokenizer(sentences, is_split_into_words=True, 
                                 truncation=True, padding="max_length", max_length=max_length)
    
    all_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Nếu là token padding
            if word_idx is None:
                label_ids.append(-100)
            # Nếu là token đầu tiên của từ, lấy nhãn gốc
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                # Chiến lược: gán nhãn -100 cho token con (để không tính loss)
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# tokenizer = AutoTokenizer.from_pretrained("vinai/phoBERT_base", use_fast=False)
# # Giả sử label2id đã được định nghĩa, ví dụ:
# label_list = ["O", "B-ADDRESS", "I-ADDRESS", "B-PHONE", "I-PHONE", "B-STORE", "I-STORE"]
# label2id = {label: idx for idx, label in enumerate(label_list)}
# id2label = {idx: label for label, idx in label2id.items()}

# tokenized_data = tokenize_and_align_labels(sentences, labels, label2id)

# # Ví dụ:
# sentences, labels = load_data("data/train.txt")


def combine_line(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()  # read all line into list
    all = lines[487:1759]
    for i in range(0, len(all), 4):
        line = f"{all[i].strip()} {all[i+1].strip()} {all[i+2].strip()}"
        with open("./data/raw_data.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

combine_line("./data/raw_data.txt")