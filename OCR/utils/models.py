


import torch


def load_abcnetv2_model():
    """
    Initialize ABCNetv2 model with specified parameters.
    """
    from OCR.detector.detectron2_handler import Detectron2TextDetector 
    from OCR.detector import prepare_cfg_detectron
    
    cfg = prepare_cfg_detectron()
    abcnet = Detectron2TextDetector(cfg)
    return abcnet
def load_paddleocr_model(lang: str = 'vi', use_angle_cls: bool = True, cls: bool = True, det: bool = True, rec: bool = False, det_db_box_thresh: float = 0.3):
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
def load_vietocr_model(config_path: str = None, device: str = None):
    """
    Initialize VietOCR model with specified parameters.
    """
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not config_path:
        config_path = 'OCR/config/vietocr.yml'
    from OCR.ocr.vietocr_handler import VietOCRWrapper
    vietocr = VietOCRWrapper(
        config_path=config_path,
        device=device
    )
    return vietocr

def load_yolov8_model(path: str = None):
    """
    Initialize YOLOv8 model with specified parameters.
    """
    if not path:
        path = "OCR/weights/signboard_model.onnx"
    from OCR.detector.yolov8_handler import YoloV8Handler
    
    model = YoloV8Handler(model_path=path)
    return model
