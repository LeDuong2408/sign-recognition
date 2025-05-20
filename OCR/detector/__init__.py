from detectron2.data.detection_utils import read_image
from adet.config import get_cfg
import os

import torch

def prepare_cfg_detectron(confidence_threshold: float = 0.3, weights: str = 'OCR/weights/abcnetv2.pth', config_file: str = 'OCR/config/BAText/TotalText/v2_attn_R_50.yaml'):
    cfg = get_cfg()
    
    config_file_path = os.path.abspath(config_file)
    weights_path = os.path.abspath(weights)
    
    cfg.merge_from_file(config_file_path)
    cfg.merge_from_list(['MODEL.WEIGHTS', weights_path])
    
    
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Set score_threshold for builtin models
    threshold = float(confidence_threshold)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold
    cfg.freeze()
    return cfg
