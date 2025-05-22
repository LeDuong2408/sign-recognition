import asyncio
from http.client import HTTPException
from io import BytesIO
import cv2
import httpx
import numpy as np
from OCR import inference_ocr
from POST_OCR import inference_ocr_correction
from NER.Using_pipeline import inference_ner

from PIL import Image
def pipline(imgs: list[Image.Image]) -> list[str]: # type: ignore
    
    ocred_texts = inference_ocr(imgs=imgs)
    
    corrected_texts = inference_ocr_correction(ocred_texts=ocred_texts)
    
    ner_result = inference_ner(corrected_texts)
    # TODO NER here ...................................................
    
    return ocred_texts,corrected_texts, ner_result