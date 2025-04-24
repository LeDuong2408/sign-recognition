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
        ent = r['entity']
        word = r['word']

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
            next_ent = ner_result[idx + 1]['entity'] if idx + 1 < len(ner_result) else None
            if next_ent != "I-LOCATION":
                address = address.rstrip() + ", "

    name_shop = name_shop.strip()
    address = address.rstrip(", ").strip()
    name_shop = merge_wordpieces(name_shop)
    address = merge_wordpieces(address)
    return f"Name: {name_shop} \nAddress: {address} \n"

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
    return ' '.join(result)
def extract_phone_number(text):
   
    pattern = r"\b\d{4}[\s\.-]?\d{4}[\s\.-]?\d{3}\b"
    match = re.search(pattern, text)
    if match:
        raw_phone = match.group(0)
        phone_number = re.sub(r"[\s\.-]", "", raw_phone)
        phone_number = phone_number.replace(" ", "")
        return phone_number
    
    pattern = r"\b\d{4}[\s\.-]?\d{3}[\s\.-]?\d{3}\b"
    match = re.search(pattern, text)
    if match:
        raw_phone = match.group(0)
        phone_number = re.sub(r"[\s\.-]", "", raw_phone)
        phone_number = phone_number.replace(" ", "")
        return phone_number
        
    
if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    
    checkpoint = torch.load("./NER/out/checkpoints/checkpoint_best.pkl", map_location="cpu")
    state_dict = checkpoint["model_state"]
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v
    model.load_state_dict(new_state)

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    example = "Quán cà phê Gò Vấp view đẹp tha hồ sống ảo Phen's Coffee - quán cafe đẹp ở Sài Gòn gần sân bay Địa chỉ: 142 Nguyễn Văn Công, Phường 3, Quận Gò Vấp, TP.HCM. 0869981478"
    begin = datetime.now()
    out =  format_out(nlp(example)) + "Tel: " + extract_phone_number(example)
    end = datetime.now()
    print(example)
    print(out)
    # print(nlp(example))
    # print("Time taken: ", (end - begin).total_seconds())