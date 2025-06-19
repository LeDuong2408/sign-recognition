import logging
from typing import Literal
import torch



def load_abcnetv2_model(weights: str = 'OCR/weights/abcnetv2.pth'):
    """
    Initialize ABCNetv2 model with specified parameters.
    """
    from OCR.detector.detectron2_handler import Detectron2TextDetector 
    from OCR.detector import prepare_cfg_detectron
    
    logging.info('Start Loading ABCNetV2....')
    cfg = prepare_cfg_detectron(weights=weights)
    abcnet = Detectron2TextDetector(cfg)
    logging.info('Done Load ABCNetV2')
    return abcnet
def load_paddleocr_model(lang: str = 'vi', use_angle_cls: bool = True, cls: bool = True, det: bool = True, rec: bool = False, det_db_box_thresh: float = 0.3):
    """
    Initialize PaddleOCR model with specified parameters.
    """
    from OCR.ocr.paddleocr_handler import PaddleOCRWrapper
    logging.info('Start Loading PADDLE....')
    
    paddle = PaddleOCRWrapper(
        lang=lang,
        use_angle_cls=use_angle_cls,
        cls=cls,
        det=det,
        rec=rec,
        det_db_box_thresh=det_db_box_thresh
    )
    logging.info('Done Load PADDLE')
    return paddle
def load_vietocr_model(weights: Literal['default','fine-tune'] = 'fine-tune',config_path: str = None, device: str = None):
    """
    Initialize VietOCR model with specified parameters.
    """
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not config_path:
        config_path = 'OCR/config/vietocr.yml'
    logging.info(f'Device set to {device}')
    from OCR.ocr.vietocr_handler import VietOCRWrapper
    logging.info('Start Loading VIETOCR....')
    vietocr = VietOCRWrapper(
        weights=weights,
        config_path=config_path,
        device=device
    )
    logging.info('Done load VIETOCR')

    return vietocr

def load_yolov8_model(path: str = None):
    """
    Initialize YOLOv8 model with specified parameters.
    """
    if not path:
        path = "OCR/weights/signboard_model.onnx"
    from OCR.detector.yolov8_handler import YoloV8Handler
    logging.info('Start Loading YOLO...')
    model = YoloV8Handler(model_path=path)
    logging.info('Done Load YOLO')
    return model
