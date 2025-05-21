from POST_OCR.correction.Gemini_handler import OCRCorrection

ocr_corrector = OCRCorrection()

def inference_ocr_correction(ocred_texts: list[str]) -> list[str]:
    result = ocr_corrector(text=ocred_texts)
    return result