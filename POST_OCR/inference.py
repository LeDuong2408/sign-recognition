from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

model = T5ForConditionalGeneration.from_pretrained("vanwdai/byt5-base-vi-ocr-correction")
tokenizer = AutoTokenizer.from_pretrained("vanwdai/byt5-base-vi-ocr-correction")

def inference_post_ocr(ocred_text: str, model: T5ForConditionalGeneration = model, tokenizer: any = tokenizer) -> str:
    
    with torch.no_grad():
        inputs = tokenizer(ocred_text, return_tensors="pt", truncation=True, max_length=256)

        output_sequences = model.generate(

            input_ids=inputs["input_ids"],

            attention_mask=inputs["attention_mask"],

            max_length=256,
            
        )

        output = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        return output