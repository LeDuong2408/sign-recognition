from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from OCR.utils.config_loader import load_config
import cv2


class VietOCRWrapper:
    def __init__(self, config_path='OCR/config/vietocr.yml', device=None):
        config = load_config(config_path)
        self.cfg = Cfg.load_config_from_file(config.config_path)
        self.cfg['weights'] = config.weights
        print('Weights:', self.cfg['weights'])
        self.cfg['device'] = device or ('cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
        self.ocr = Predictor(self.cfg)

    def __call__(self, image):
        return self.ocr.predict(image)