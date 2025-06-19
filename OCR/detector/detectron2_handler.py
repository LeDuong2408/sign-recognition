import torch
from detectron2.engine import DefaultPredictor


class Detectron2TextDetector:
    def __init__(self, cfg, threshold_score=0.5):
        self.predictor = DefaultPredictor(cfg)
        self.threshold_score = threshold_score

    def detect(self, image, rt_beziers=False):
        predictions = self.predictor(image)
        instances = predictions["instances"]
        beziers = instances.beziers.cpu().detach().tolist()
        scores = instances.scores.tolist()
        recs = instances.recs
        boxes, box_scores, box_beziers = [], [], []
        for p, rec, score in zip(beziers, recs, scores):
            if score < self.threshold_score:
                continue
            p = [list(map(int, p[i : i + 2])) for i in range(0, len(p), 2)]
            box = [p[0], p[3], p[4], p[7]]
            boxes.append(box)
            box_scores.append(score)
            box_beziers.append(p)
        if rt_beziers:
            return boxes, box_scores, box_beziers
        return boxes, box_scores

    def __call__(self, image, rt_beziers=False):
        return self.detect(image=image, rt_beziers=rt_beziers)
