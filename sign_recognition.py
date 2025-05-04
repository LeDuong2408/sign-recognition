from  OCR.vietocr_sign import get_ocr_text
from NER.Using_pipeline import get_ner_result
from POST_OCR.inference import inference_post_ocr
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, T5ForConditionalGeneration
import torch
from OCR.config import PATH_IMAGE

if __name__ == "__main__":
    # Load the Post-OCR model
    # model_post_ocr = T5ForConditionalGeneration.from_pretrained("vanwdai/byt5-base-vi-ocr-correction")
    # tokenizer_post_ocr = AutoTokenizer.from_pretrained("vanwdai/byt5-base-vi-ocr-correction")
   
    # Load the NER model
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
    with torch.no_grad():
        nlp.model.eval()
        image_path = PATH_IMAGE
        ocr_texts = get_ocr_text(image_path)
        print("OCR Text: ", ocr_texts)
        ocr_texts = ' '.join(ocr_texts)

        post_ocr_text = inference_post_ocr(ocr_texts)
        print("Post OCR Text", post_ocr_text)

        ner_result = get_ner_result(post_ocr_text.upper(), nlp)
        print("NER results: ", ner_result)

        