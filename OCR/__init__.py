import logging
import os

import concurrent
from OCR.utils.image_utils import perspective_transform, preprocess_crop_img
from OCR.utils.line_assignment import assign_words_to_lines, reorder_text
from OCR.utils.models import (
    load_abcnetv2_model,
    load_vietocr_model,
    load_yolov8_model,
    load_paddleocr_model,
)
from PIL import Image

abcnet_text_det = load_abcnetv2_model()
paddleocr_line_det = load_paddleocr_model()
vietocr_tex_rec = load_vietocr_model()
yolov8_sign_det = load_yolov8_model()


def inference_ocr(imgs: list[Image.Image]) -> list[str]:  # type: ignore
    logging.info("Start inference ocr...")
    texts_ocr: list[str] = []

    for id, img in enumerate(imgs):
        if img is None:
            logging.warning(f"Cannot read image or image is invalid: {id}")
            continue

        # Step 1: rotate image
        img_rotated = paddleocr_line_det.get_rotated_image(img)

        boxes_signboard = yolov8_sign_det.get_boxes_best_score(img_rotated)

        if boxes_signboard is None:
            logging.error(f"[ERROR YOLO] Cannot detect signboard in image: {id}")
            image_signboard = img_rotated.copy()
        else:
            image_signboard = perspective_transform(img_rotated, boxes_signboard[0])

        # Step 2: detect text lines
        # boxes_line = paddleocr_line_det.get_boxes_line(image_signboard)

        # Step 3: detect each box words
        boxes_word, _ = abcnet_text_det(image=image_signboard)

        texts = []

        for i, box in enumerate(boxes_word):
            crop_img = perspective_transform(image_signboard, box)
            crop_img = preprocess_crop_img(crop_img)

            try:
                text = vietocr_tex_rec(crop_img)
                texts.append(text)
            except Exception as e:
                logging.error(f"Error VIETOCR processing box {i}: {e}")
                texts.append("")
        # Step 4: assign words to lines
        # word_each_lines = assign_words_to_lines(boxes_word, boxes_line, texts, image_signboard.shape)
        # list_all_word = [word[1] for line in word_each_lines for word in line]
        # final_text = ' '.join(list_all_word)
        boxes_word, texts_reoder = reorder_text(boxes_word, texts)
        setence = " ".join(texts_reoder)
        texts_ocr.append(setence)
    logging.info("Done inference ocr")
    return texts_ocr


# def process_rotated_image(id: int, img_rotated: Image.Image) -> str:
#     try:
#         boxes_signboard = yolov8_sign_det.get_boxes_best_score(img_rotated)

#         if boxes_signboard is None:
#             logging.error(f"[YOLO] Cannot detect signboard in image: {id}")
#             image_signboard = img_rotated.copy()
#         else:
#             image_signboard = perspective_transform(img_rotated, boxes_signboard[0])

#         boxes_word, _ = abcnet_text_det(image=image_signboard)

#         texts = []
#         for i, box in enumerate(boxes_word):
#             crop_img = perspective_transform(image_signboard, box)
#             crop_img = preprocess_crop_img(crop_img)

#             try:
#                 text = vietocr_tex_rec(crop_img)
#                 texts.append(text)
#             except Exception as e:
#                 logging.error(f"[VIETOCR] Error at box {i} (image {id}): {e}")
#                 texts.append("")

#         boxes_word, texts_reorder = reorder_text(boxes_word, texts)
#         return " ".join(texts_reorder)

#     except Exception as e:
#         logging.error(f"[OCR] Unexpected error in image {id}: {e}")
#         return ""


# # Hàm chính để xử lý nhiều ảnh
# def inference_ocr(imgs: list[Image.Image]) -> list[str]:
#     logging.info("Start inference ocr...")
#     texts_ocr = [""] * len(imgs)

#     # Step 1: Xoay ảnh bằng PaddleOCR (tuần tự)
#     preprocessed = []
#     for i, img in enumerate(imgs):
#         try:
#             if img is None:
#                 logging.warning(f"[Preprocess] Invalid image at index {i}")
#                 continue
#             img_rotated = paddleocr_line_det.get_rotated_image(img)
#             preprocessed.append((i, img_rotated))
#         except Exception as e:
#             logging.error(f"[PaddleOCR] Error rotating image {i}: {e}")
#             continue

#     # Step 2: Song song xử lý phần còn lại
#     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#         futures = {
#             executor.submit(process_rotated_image, i, img): i for i, img in preprocessed
#         }

#         for future in concurrent.futures.as_completed(futures):
#             i = futures[future]
#             try:
#                 result = future.result()
#                 logging.info(f"[OCR] Processed image {i} successfully. {result}")
#                 texts_ocr[i] = result
#             except Exception as e:
#                 logging.error(f"[OCR] Error in thread for image {i}: {e}")
#                 texts_ocr[i] = ""

#     logging.info("Done inference ocr")
#     return texts_ocr
