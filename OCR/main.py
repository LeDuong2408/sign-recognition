import argparse
import os
import glob
import sys
import time

import cv2

from OCR.detector import prepare_cfg_detectron, read_image
from OCR.detector.detectron2_handler import Detectron2TextDetector 

from OCR.detector.yolov8_handler import YoloV8Handler
from OCR.ocr.paddleocr_handler import PaddleOCRWrapper

from OCR.ocr.vietocr_handler import VietOCRWrapper
from OCR.utils.image_utils import perspective_transform, preprocess_crop_img
from OCR.utils.line_assignment import assign_words_to_lines
from OCR.utils.visualizer import visualize_detection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image or directory")
    parser.add_argument("--output", default="output", help="Path to output directory")
    # parser.add_argument("--confidence_threshold", default=0.4)
    # parser.add_argument("--weights", required=True, help="Path to model ABCnetv2 weights")
    # parser.add_argument(
    #     "--opts",
    #     help="Modify config options using the command-line 'KEY VALUE' pairs",
    #     default=[],
    #     nargs=argparse.REMAINDER,
    # )
    # parser.add_argument(
    #     "--config-file",
    #     default="OCR/config/BAText/TotalText/v2_attn_R_50.yaml",
    #     metavar="FILE",
    #     help="path to config file",
    # )
    return parser

def get_image_files(input_path: str) -> list: # type: ignore
    """
    Get all image files from the input directory.
    """
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if os.path.isdir(input_path):
        files = [f for f in glob.glob(os.path.join(input_path, "*.*")) if f.lower().endswith(extensions)]
        return files
    
    elif os.path.isfile(input_path):
        if input_path.lower().endswith(extensions):
            return [input_path]
        return []
    
    else:
        print(f"Input path {input_path} is not a directory.")
        return []

def load_abcnetv2_model():
    """
    Initialize ABCNetv2 model with specified parameters.
    """
    from OCR.detector.detectron2_handler import Detectron2TextDetector 
    cfg = prepare_cfg_detectron()
    abcnet = Detectron2TextDetector(cfg)
    return abcnet
def paddleocr_model(lang: str, use_angle_cls: bool = True, cls: bool = True, det: bool = True, rec: bool = False, det_db_box_thresh: float = 0.3):
    """
    Initialize PaddleOCR model with specified parameters.
    """
    from OCR.ocr.paddleocr_handler import PaddleOCRWrapper
    
    paddle = PaddleOCRWrapper(
        lang=lang,
        use_angle_cls=use_angle_cls,
        cls=cls,
        det=det,
        rec=rec,
        det_db_box_thresh=det_db_box_thresh
    )
    return paddle
def load_vietocr_model(config_path: str, device: str):
    """
    Initialize VietOCR model with specified parameters.
    """
    from OCR.ocr.vietocr_handler import VietOCRWrapper
    vietocr = VietOCRWrapper(
        config_path=config_path,
        device=device
    )
    return vietocr

def load_yolov8_model(path: str):
    """
    Initialize YOLOv8 model with specified parameters.
    """
    model = YoloV8Handler(model_path=path)
    return model

def main():
    args = get_parser().parse_args()
    cfg = prepare_cfg_detectron()
    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    os.makedirs(output_path, exist_ok=True)

    detectron_abcnet = Detectron2TextDetector(cfg)

    paddle = PaddleOCRWrapper(
        use_gpu=(device == 'cuda'),
        lang='vi',
        use_angle_cls=True,
        cls=True,
        det=True,
        rec=False,
        det_db_box_thresh=0.3
    )
    
    vietocr = VietOCRWrapper(
        config_path='OCR/config/vietocr.yml',
        device=device
    )
    
    yolov8 = load_yolov8_model("OCR/weights/signboard_model.onnx")
    
    print('Ready for predictions!')
    
    files = get_image_files(input_path)
    if not files:
        print(f"No image files found in {input_path}.")
        return
    print(f"Found {len(files)} image files.")
    trace = {}
    start_time = time.time()
    
    for id, img_path in enumerate(files):
        img = read_image(img_path, format="BGR")
        if img is None:
            print(f"Cannot read image or image is invalid: {img_path}")
            continue
        filename = os.path.basename(img_path)
        filename_noext = filename.split(".")[0]
        
        # Step 1: rotate image
        img_rotated = paddle.get_rotated_image(img)

        boxes_signboard = yolov8.get_boxes_best_score(img_rotated)
        
        if boxes_signboard is None:
            print(f"[ERROR YOLO] Cannot detect signboard in image: {img_path}")
            image_signboard = img_rotated.copy()
        else:
            visualize_detection(img_rotated, boxes_signboard, f"{filename_noext}_signboard", output_path)
            image_signboard =  perspective_transform(img_rotated, boxes_signboard[0])
        
        # Step 2: detect text lines
        boxes_line = paddle.get_boxes_line(image_signboard)
        # Step 3: detect each box words
        boxes_word, box_scores = detectron_abcnet(image_signboard)
        
        texts = []
        
        for i, box in enumerate(boxes_word):
            crop_img = perspective_transform(image_signboard, box)
            crop_img = preprocess_crop_img(crop_img)
            
            try:
                text = vietocr(crop_img)
                texts.append(text)
            except Exception as e:
                print(f"Error processing box {i}: {e}")
                texts.append("")
        # Step 4: assign words to lines
        lines = assign_words_to_lines(boxes_word, boxes_line, texts, image_signboard.shape)
        
        if filename in trace:
            print("Duplicated filename:", filename)
            break
        trace[filename] = 1

        # with open(os.path.join(output_path, f"{filename_noext}_score.txt"), 'w', encoding="utf-8") as f:
        #     for box, score in zip(boxes_word, box_scores):
        #         points_str = ','.join(f"{x},{y}" for x, y in box)
        #         f.write(f"{points_str},{score}\n")
        
        with open(os.path.join(output_path, f"{filename_noext}.txt"), "w", encoding="utf-8") as f:
            line_text = [word[1] for line in lines for word in line]
            f.write(' | '.join(line_text))
        
        visualize_detection(image_signboard, boxes_word, filename_noext, output_path)
        visualize_detection(image_signboard, boxes_line, f'{filename_noext}_line', output_path)
        
        print(f"[✔] Done {filename} - {id+1}/{len(files)}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
            
    