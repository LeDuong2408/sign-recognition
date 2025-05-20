import cv2
import numpy as np
import torch
import onnxruntime as ort

class YoloV8Handler:
    def __init__(self, model_path: str, conf_thres=0.3, iou_thres=0.5):
        """
        Initialize the YoloV8Handler with the path to the YOLOv8 model.

        Args:
            model_path (str): Path to the YOLOv8 model file.
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.model_path = model_path
        # self.model = self.load_model()
        self._initialize_model(path=model_path)

    def _initialize_model(self, path):
        self.session = ort.InferenceSession(
            path,
            # providers=ort.get_available_providers()
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        self._get_input_details()
        self._get_output_details()
    
    def _get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def _get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    
    def detect_objects(self, image):
        img = self._prepare_input(image)
        
        outputs = self._predict(img)
        boxes, scores, class_ids = self.process_output(outputs)
        return boxes, scores, class_ids
    
    def get_img_resized(self):
        return self.image_resized
        
    def _prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_resized  = self._letterbox(input_img, new_shape=(self.input_width, self.input_height))
        # Resize input image
        # img_resized = cv2.resize(input_img, (self.input_width, self.input_height))
        img_return = img_resized.copy()
        img_return = cv2.cvtColor(img_return, cv2.COLOR_RGB2BGR)
        self.image_resized = img_return
        # Scale input pixel values to 0 to 1
        img = img_resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)

        return img

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        Resize ảnh với padding để giữ nguyên tỉ lệ (không méo).
        - `auto`: tự điều chỉnh kích thước padding thành bội số của 32 (để dùng cho YOLO)
        - `scaleFill`: ép hình ảnh để vừa khít khung (có thể gây méo)
        - `scaleup`: cho phép phóng to ảnh nhỏ (nếu False sẽ không phóng lên)

        Returns:
            img_resized: ảnh đã resize
            ratio: (scale_w, scale_h)
            (dw, dh): padding đã thêm theo chiều rộng và cao
        """
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Compute scale ratio (new / old) and padding
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        
        self.ratio = (r, r)

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width, height padding

        if auto:
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # make divisible by 32

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        self.dw = dw
        self.dh = dh
        # Resize
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img_padded
    
    def _predict(self, img):
        outputs = self.session.run(self.output_names, {self.input_names[0]: img})

        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = self.multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    
    def multiclass_nms(self, boxes, scores, class_ids, iou_threshold):
        def compute_iou(box, boxes):
            # Compute xmin, ymin, xmax, ymax for both boxes
            xmin = np.maximum(box[0], boxes[:, 0])
            ymin = np.maximum(box[1], boxes[:, 1])
            xmax = np.minimum(box[2], boxes[:, 2])
            ymax = np.minimum(box[3], boxes[:, 3])

            # Compute intersection area
            intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

            # Compute union area
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            union_area = box_area + boxes_area - intersection_area

            # Compute IoU
            iou = intersection_area / union_area

            return iou
        
        def nms(boxes, scores, iou_threshold):
            # Sort by score
            sorted_indices = np.argsort(scores)[::-1]

            keep_boxes = []
            while sorted_indices.size > 0:
                # Pick the last box
                box_id = sorted_indices[0]
                keep_boxes.append(box_id)

                # Compute IoU of the picked box with the rest
                ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

                # Remove boxes with IoU over the threshold
                keep_indices = np.where(ious < iou_threshold)[0]

                # print(keep_indices.shape, sorted_indices.shape)
                sorted_indices = sorted_indices[keep_indices + 1]

            return keep_boxes
        
        unique_class_ids = np.unique(class_ids)

        keep_boxes = []
        for class_id in unique_class_ids:
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes[class_indices,:]
            class_scores = scores[class_indices]

            class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
            keep_boxes.extend(class_indices[class_keep_boxes])

        return keep_boxes

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        
        boxes = self.xywh2xyxy(boxes)
        
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        """
        Chuyển bounding boxes từ ảnh đã resize + padding về tọa độ gốc ban đầu của ảnh.

        Args:
            boxes (ndarray): Mảng (N, 4) với bbox theo dạng [x1, y1, x2, y2] trên ảnh đã padding.
            ratio (tuple): (scale_x, scale_y) tỉ lệ resize của ảnh gốc.
            dw (float): padding theo chiều rộng đã chia đôi (left/right).
            dh (float): padding theo chiều cao đã chia đôi (top/bottom).

        Returns:
            ndarray: Mảng (N, 4) bbox theo ảnh gốc (không resize/padding).
        """
        boxes = boxes.copy()
        # Trừ padding
        boxes[:, [0, 2]] -= self.dw
        boxes[:, [1, 3]] -= self.dh

        # Chia lại theo tỷ lệ scale
        boxes[:, [0, 2]] /= self.ratio[0]
        boxes[:, [1, 3]] /= self.ratio[1]

        # Clip to image size (optional, tránh vượt giới hạn ảnh gốc)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.img_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.img_height)

        return boxes

    def xywh2xyxy(self, boxes):
        """
        Chuyển các bounding boxes từ định dạng (x_center, y_center, w, h)
        sang định dạng (x1, y1, x2, y2)

        Args:
            boxes (ndarray): Mảng numpy (N x 4) với bbox theo dạng [x_center, y_center, width, height]

        Returns:
            ndarray: Mảng (N x 4) theo dạng [x1, y1, x2, y2]
        """
        boxes = boxes.copy()
        x_c = boxes[:, 0]
        y_c = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2

        return np.stack([x1, y1, x2, y2], axis=1)
    
    def get_boxes_best_score(self, image):
        """
        Get the bounding boxes with the best score from the detected objects.
        Args:
            image (numpy.ndarray): Input image in BGR format.
        Returns:
            boxes (list): List of detected bounding boxes with the best score.
        """

        boxes, scores, class_ids = self.detect_objects(image)

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
        return self.detect_objects(image)
    