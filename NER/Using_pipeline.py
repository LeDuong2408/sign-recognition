import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from datetime import datetime
import re
import torch


def format_out(ner_result):
    name_shop = ""
    address = ""
    flag = None

    for idx, r in enumerate(ner_result):
        ent = r["entity"]
        word = r["word"]

        if ent == "B-ORGANIZATION":
            name_shop += word + " "
            flag = "ORGANIZATION"
        elif flag == "ORGANIZATION" and ent == "I-ORGANIZATION":
            name_shop += word + " "

        if ent == "B-LOCATION":
            address += word + " "
            flag = "LOCATION"
        elif flag == "LOCATION" and ent == "I-LOCATION":
            address += word + " "
            next_ent = (
                ner_result[idx + 1]["entity"] if idx + 1 < len(ner_result) else None
            )
            if next_ent != "I-LOCATION":
                address = address.rstrip() + ", "

    name_shop = name_shop.strip()
    address = address.rstrip(", ").strip()
    name_shop = merge_wordpieces(name_shop)
    address = merge_wordpieces(address)
    return {
        "name": name_shop,
        "address": address,
    }


def merge_wordpieces(text):
    tokens = text.split()
    result = []
    for token in tokens:
        if token.startswith("##"):
            if result:
                result[-1] += token[2:]  # nối phần sau "##" vào từ trước
            else:
                result.append(token[2:])  # phòng khi không có từ trước
        else:
            result.append(token)
    return " ".join(result)


# def extract_phone_number(text):
#     NUMERIC_PATTERN = r"\b\d{3,4}(?:[\s\.-]?\d{3,4}){2}\b"
#     PREFIX_PATTERN = r"\b(?:SDT|SĐT|PHONE|TEL)[\s:\.-]*(\d{3,4}(?:[\s\.-]?\d{3,4}){2})\b"

#     matches_numeric = re.findall(NUMERIC_PATTERN, text, flags=re.IGNORECASE)
#     matches_prefix = re.findall(PREFIX_PATTERN, text, flags=re.IGNORECASE)

#     all_matches = matches_numeric + matches_prefix
#     cleaned = [re.sub(r"[\s\.-]", "", m) for m in all_matches]

#     unique_numbers = " - ".join(cleaned)

#     return unique_numbers if unique_numbers else None


def extract_phone_number(text):
    def clean_number(number: str) -> str:
        # Loại bỏ mọi ký tự không phải số, trừ dấu '+' ở đầu nếu có
        number = number.strip()
        return re.sub(r"[^\d+]", "", number)

    def apply_special_rule(num: str) -> str:
        digits_only = re.sub(r"\D", "", num)
        # Nếu là số có đúng 9 chữ số và chứa '77', đổi '77' thành '777'
        if len(digits_only) == 9 and "77" in digits_only:
            digits_only = digits_only.replace("77", "777", 1)
            return digits_only
        return num

    # Regex: nhận các số bắt đầu bằng +84, 84 hoặc 0, dài từ 9-11 số, phân tách bởi space/dot/dash
    PREFIX_PATTERN = (
        r"\b(?:SDT|SĐT|PHONE|TEL)[\s:\.-]*((?:\+?84|0)(?:[\s\.-]*\d){8,10})(?!\d)"
    )
    GENERIC_PATTERN = r"(?<!\d)((?:\+?84|0)(?:[\s\.-]*\d){8,10})(?!\d)"

    # Tìm tất cả khớp
    matches_prefix = re.findall(PREFIX_PATTERN, text, flags=re.IGNORECASE)
    matches_generic = re.findall(GENERIC_PATTERN, text, flags=re.IGNORECASE)

    # Gộp và làm sạch
    raw_matches = matches_prefix + matches_generic
    cleaned = [clean_number(m) for m in raw_matches]

    # Giữ lại số hợp lệ: tổng số chữ số từ 9 đến 13 (cho phép +84 ở đầu)
    validated = [m for m in cleaned if 9 <= len(re.sub(r"\D", "", m)) <= 13]

    # Áp dụng quy tắc đặc biệt
    final = [apply_special_rule(m) for m in validated]

    # Loại trùng và trả kết quả
    unique_numbers = " - ".join((set(final)))
    return unique_numbers if unique_numbers else None


def load_model_ner():
    tokenizer = AutoTokenizer.from_pretrained("duongai248/ner-location-vietnam")
    model = AutoModelForTokenClassification.from_pretrained(
        "duongai248/ner-location-vietnam"
    )
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    return nlp


nlp = load_model_ner()


def get_ner_result(text, nlp):
    phone = extract_phone_number(text)
    if phone:
        out = format_out(nlp(text))
        out["tel"] = phone
    else:
        out = format_out(nlp(text))
        out["tel"] = ""
    return out


# TODO =============================================================
def clean_text(text):
    phrases_to_remove = ["Coca Cola NGHỈ XẢ HƠI", "Bia Việt"]
    for phrase in phrases_to_remove:
        if phrase in text:
            text = text.replace(phrase, "")
    return text.strip()


def inference_ner(ocred_texts: list[str]) -> list[dict]:
    logging.info("Starting inference NER...")
    res = []
    for text in ocred_texts:
        text = clean_text(text)
        result = get_ner_result(text=text.title(), nlp=nlp)
        res.append(result)
    logging.info("Done inference NER")
    return res


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    model = AutoModelForTokenClassification.from_pretrained(
        "NlpHUST/ner-vietnamese-electra-base"
    )

    checkpoint = torch.load(
        "./NER/out/checkpoints/checkpoint_best.pkl", map_location="cpu"
    )
    state_dict = checkpoint["model_state"]
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v
    model.load_state_dict(new_state)

    # nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    # example = "tạp hóa phú mỹ, 621 đường huỲnh văn luỲ p. phú mỸ tp. thủ đầu mỘt bd, SDT0908 123 456"
    # # example = example.upper()
    # begin = datetime.now()
    # out =  get_ner_result(example, nlp)
    # end = datetime.now()
    # print(example)
    # print(out)

    # print(nlp(example))
    # print("Time taken: ", (end - begin).total_seconds())

    model.push_to_hub("duongai248/ner-location-vietnam")
    tokenizer.push_to_hub("duongai248/ner-location-vietnam")

    # from huggingface_hub import HfApi
    # import os
    # api = HfApi(token=os.getenv("HF_TOKEN"))
    # api.upload_folder(
    #     folder_path= "F:/University_HCMUTE/N4_HK2/KLTN/Sign_Recognition/NER/data/electra-vn",
    #     repo_id="duongai248/ner-location-vietnam",
    #     repo_type="dataset",
    # )
