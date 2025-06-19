import logging
from POST_OCR.correction.Gemini_handler import OCRCorrection

ocr_corrector = OCRCorrection()


def inference_ocr_correction(ocred_texts: list[str]) -> list[str]:
    logging.info("Starting inference POSTOCR...")
    result = ocr_corrector(text=ocred_texts)
    logging.info("Done inference POSTOCR")
    return result
