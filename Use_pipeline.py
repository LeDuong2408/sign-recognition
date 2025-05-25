from pipeline import pipeline
from PIL import Image
import numpy as np

def load_image(source):
    if isinstance(source, str):
        # if source.startswith("http"):
        #     return asyncio.run(fetch_image(source))  
        # else:
        return np.array(Image.open(source).convert("RGB"))
    return source 

if __name__ == '__main__':
    image_paths = ["./OCR/data/bien_hieu3.png", "./OCR/data/bien_hieu2.png"] 
    images = [load_image(path) for path in image_paths]
    
    ocred_texts, corrected_texts, ner_result = pipeline(images)
    
    # print("OCRed Texts:", ocred_texts)
    # print("Corrected Texts:", corrected_texts)
    print("NER Result:", ner_result)