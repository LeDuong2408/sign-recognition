import os
from OCR.utils.image_utils import perspective_transform, preprocess_crop_img
from OCR.utils.line_assignment import assign_words_to_lines
from OCR.utils.models import load_abcnetv2_model, load_vietocr_model, load_yolov8_model, load_paddleocr_model
from PIL import Image

abcnet_text_det = load_abcnetv2_model()
paddleocr_line_det = load_paddleocr_model()
vietocr_tex_rec = load_vietocr_model()
yolov8_sign_det = load_yolov8_model()

def inference_ocr(imgs: list[Image.Image]) -> list[str]:
    
    texts_ocr: list[str] = []
    
    for id, img in enumerate(imgs):
        if img is None:
            print(f"Cannot read image or image is invalid: {id}")
            continue
        
        # Step 1: rotate image
        img_rotated = paddleocr_line_det.get_rotated_image(img)

        boxes_signboard = yolov8_sign_det.get_boxes_best_score(img_rotated)
        
        if boxes_signboard is None:
            print(f"[ERROR YOLO] Cannot detect signboard in image: {id}")
            image_signboard = img_rotated.copy()
        else:
            image_signboard =  perspective_transform(img_rotated, boxes_signboard[0])
        
        # Step 2: detect text lines
        boxes_line = paddleocr_line_det.get_boxes_line(image_signboard)
        # Step 3: detect each box words
        boxes_word, box_scores = abcnet_text_det(image_signboard)
        
        texts = []
        
        for i, box in enumerate(boxes_word):
            crop_img = perspective_transform(image_signboard, box)
            crop_img = preprocess_crop_img(crop_img)
            
            try:
                text = vietocr_tex_rec(crop_img)
                texts.append(text)
            except Exception as e:
                print(f"Error processing box {i}: {e}")
                texts.append("")
        # Step 4: assign words to lines
        word_each_lines = assign_words_to_lines(boxes_word, boxes_line, texts, image_signboard.shape)
        list_all_word = [word[1] for line in word_each_lines for word in line]
        final_text = ' '.join(list_all_word)
        texts_ocr.append(final_text)
    return texts_ocr