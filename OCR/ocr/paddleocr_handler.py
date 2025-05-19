import cv2
import numpy as np
from paddleocr import PaddleOCR

from OCR.utils.image_utils import pad_bbox


class PaddleOCRWrapper:
    def __init__(self, lang='en', use_gpu=False, use_angle_cls=True, det_db_box_thresh=0.3, cls=True, det=True, rec=False):
        self.cls = cls
        self.det = det
        self.rec = rec
        self.ocr = PaddleOCR(use_gpu=use_gpu, use_angle_cls=use_angle_cls, lang=lang, det_db_box_thresh=det_db_box_thresh)

    def detect(self, image):
        return self.ocr.ocr(image, cls=self.cls, det=self.det, rec=self.rec)

    def get_boxes_line(self, image):
        result = self.ocr.ocr(image, cls=self.cls, det=self.det, rec=False)
        boxes = []
        for line in result[0]:
                box = pad_bbox(np.array(line), 2, image.shape)
                boxes.append(box)
        if boxes:
            boxes = boxes[::-1]        
        
        return boxes
    
    def get_rotated_image(self, image):
        result = self.ocr.ocr(image, cls=True, det=True, rec=False)

        angles = []
        for line in result[0]:
                points = np.array(line, dtype=np.float32)
                x1, y1 = points[0]
                x2, y2 = points[1]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

        if angles:
            avg_angle = np.mean(angles)
        else:
            return image 

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def __call__(self, image):
        return self.detect(image)