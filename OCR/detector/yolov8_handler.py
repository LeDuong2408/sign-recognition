

import numpy as np


class YoloV8Handler:
    def __init__(self, model_path: str):
        """
        Initialize the YoloV8Handler with the path to the YOLOv8 model.

        Args:
            model_path (str): Path to the YOLOv8 model file.
        """
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """
        Load the YOLOv8 model from the specified path.

        Returns:
            Model: Loaded YOLOv8 model.
        """
        from ultralytics import YOLO
        model = YOLO(self.model_path)
        return model
    def detect(self, image):
        """
        Detect objects in the given image using the YOLOv8 model.
        Args:
            image (numpy.ndarray): Input image in BGR format.
        Returns:
            boxes (list): List of detected bounding boxes.
            scores (list): List of confidence scores for each detected box.
            class_ids (list): List of class IDs for each detected box.
        """
        results = self.model(image)
        boxes = []
        scores = []
        class_ids = []
        for result in results:
            if result.boxes is not None:
                class_ids = list(result.boxes.cls.cpu().numpy().astype(int).tolist())
                scores = list(result.boxes.conf.cpu().numpy().astype(float).tolist())
                boxes = list(result.boxes.xyxy.cpu().numpy().astype(int).tolist())

        return boxes, scores, class_ids
    
    def get_boxes_best_score(self, image):
        """
        Get the bounding boxes with the best score from the detected objects.
        Args:
            image (numpy.ndarray): Input image in BGR format.
        Returns:
            boxes (list): List of detected bounding boxes with the best score.
        """
        # results = self.model(image)

        # for result in results:
        #     if result.boxes is not None:
        #         class_ids = result.boxes.cls.cpu().numpy()
        #         if class_ids != 0:
        #             continue
        #         scores = result.boxes.conf.cpu().numpy()
        #         xyxy = result.boxes.xyxy.cpu().numpy()
        #         best_index = np.argmax(scores)
        #         best_box = self.xyxy2xyxyxyxy(xyxy[best_index])
        #         return best_box

        boxes, scores, class_ids = self.detect(image)

        best_index = -1
        best_score = float('-inf')

        for idx, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):

            if class_id == 0:
                if score > best_score:
                    best_score = score
                    best_index = idx
            
        if best_index != -1:
            best_box = self.xyxy2xyxyxyxy(boxes[best_index])
            return [best_box]

        return None

    def xyxy2xyxyxyxy(self, box):
        """
        Convert bounding boxes from xyxy format to xyxyxyxy format.
        Args:
            boxes (list): List of bounding boxes in xyxy format.
        Returns:
            list: List of bounding boxes in xyxyxyxy format.
        """
        x1, y1, x3, y3 = box
        x2 = x3
        y2 = y1
        x4 = x1
        y4 = y3
        return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    def __call__(self, image):
        return self.detect(image)