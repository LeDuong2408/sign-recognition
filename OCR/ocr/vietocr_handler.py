import logging
from typing import Literal
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from OCR.utils.config_loader import load_config
import torch


class VietOCRWrapper:
    def __init__(self, weights: Literal['default','fine-tune'] = 'fine-tune' ,config_path='OCR/config/vietocr.yml', device=None):
        config = load_config(config_path)
        self.cfg = Cfg.load_config_from_file(config.config_path)
        print('WEIGHTS VIETOCR: ', weights)
        if weights == 'fine-tune':
            logging.info('VIETOCR fine-tune loading...')
            print('VIETOCR fine-tune loading...')
            self.cfg['weights'] = config.weights
        self.cfg['device'] = device or ('cuda' if torch.cuda.is_available() > 0 else 'cpu')
        self.ocr = Predictor(self.cfg)

    def __call__(self, image):
        return self.ocr.predict(image)
    
    def predict_batch(self, imgs):
        self.ocr.predict_batch(imgs=imgs)