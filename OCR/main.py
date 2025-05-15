import argparse
import os
import glob
import sys
import time

import cv2

from OCR.detector import prepare_cfg_detectron, read_image
from OCR.detector.detectron2_handler import Detectron2TextDetector 

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
    parser.add_argument("--confidence_threshold", default=0.4)
    parser.add_argument("--weights", required=True, help="Path to model ABCnetv2 weights")
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--config-file",
        default="OCR/config/BAText/TotalText/v2_attn_R_50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    return parser

def get_image_files(input_path: str) -> list:
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

def main():
    args = get_parser().parse_args()
    cfg = prepare_cfg_detectron(args)
    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    os.makedirs(output_path, exist_ok=True)

    detectron = Detectron2TextDetector(cfg)

    paddle = PaddleOCRWrapper(
        lang='vi',
        use_angle_cls=True,
        cls=True,
        det=True,
        rec=False,
        det_db_box_thresh=0.3
    )
    
    vietocr = VietOCRWrapper(
        config_path='OCR/config/vietocr.yaml',
        device=device
    )
    
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
        img_rotate = paddle.get_rotated_image(img)

        # Step 2: detect text lines
        boxes_line = paddle.get_boxes_line(img_rotate)

        # Step 3: detect each box words
        boxes_word, box_scores = detectron(img_rotate)
        
        texts = []
        
        for i, box in enumerate(boxes_word):
            crop_img = perspective_transform(img_rotate, box)
            crop_img = preprocess_crop_img(crop_img)
            
            try:
                text = vietocr(crop_img)
                texts.append(text)
            except Exception as e:
                print(f"Error processing box {i}: {e}")
                texts.append("")
        # Step 4: assign words to lines
        lines = assign_words_to_lines(boxes_word, boxes_line, texts, img_rotate.shape)
        
        if filename in trace:
            print("Duplicated filename:", filename)
            break
        trace[filename] = 1

        # with open(os.path.join(output_path, f"{filename_noext}.txt"), 'w', encoding="utf-8") as f:
        #     for box, score in zip(boxes_word, box_scores):
        #         points_str = ','.join(f"{x},{y}" for x, y in box)
        #         f.write(f"{points_str},{score}\n")
        
        with open(os.path.join(output_path, f"{filename_noext}.txt"), "w", encoding="utf-8") as f:
            line_text = [word[1] for line in lines for word in line]
            f.write(' | '.join(line_text))
        
        visualize_detection(img_rotate, boxes_word, filename_noext, output_path)
        visualize_detection(img_rotate, boxes_line, f'{filename_noext}_line', output_path)
        
        print(f"[✔] Done {filename} - {id+1}/{len(files)}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
            
    