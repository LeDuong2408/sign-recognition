from detectron2.data.detection_utils import read_image
from adet.config import get_cfg
import os

def prepare_cfg_detectron(args):
    cfg = get_cfg()
    
    config_file_path = os.path.abspath(args.config_file)
    weights_path = os.path.abspath(args.weights)
    
    cfg.merge_from_file(config_file_path)
    cfg.merge_from_list(['MODEL.WEIGHTS', weights_path])
    
    # Set score_threshold for builtin models
    threshold = float(args.confidence_threshold)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold
    cfg.freeze()
    return cfg
