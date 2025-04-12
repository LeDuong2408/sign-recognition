import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import matplotlib.gridspec as gridspec

def load_vietocr_model():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = 'vgg_transformer.pth'
    config['device'] = 'cuda'  # hoặc 'cpu'
    return Predictor(config)

def auto_rotate_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Không thể đọc ảnh.")
        return None, None

    ocr = PaddleOCR(use_angle_cls=True, lang='vi')
    result = ocr.ocr(image_path, cls=True)

    angles = []
    for line in result:
        for word in line:
            points = np.array(word[0], dtype=np.float32)
            x1, y1 = points[0]
            x2, y2 = points[1]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

    if angles:
        avg_angle = np.mean(angles)
        print(f"📐 Góc xoay trung bình: {avg_angle:.2f}°")
    else:
        return image, image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return image, rotated

def detect_text_with_vietocr(image, vietocr_model):
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)

    ocr = PaddleOCR(use_angle_cls=True, lang="vi", det_db_box_thresh=0.3)
    result = ocr.ocr(sharpened, cls=True)

    binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    recognized_texts = []

    for line in result:
        for word in line:
            points = np.array(word[0], dtype=np.int32)
            cv2.fillPoly(binary_mask, [points], 255)

            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)

            x_min = max(0, x_min - 10)
            y_min = max(0, y_min - 10)
            x_max = min(image.shape[1], x_max + 10)
            y_max = min(image.shape[0], y_max + 10)

            roi = sharpened[y_min:y_max, x_min:x_max]
            if roi.shape[0] < 10 or roi.shape[1] < 10:
                recognized_texts.append("[ROI nhỏ/bị lỗi]")
                continue

            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_pil = Image.fromarray(roi_rgb).resize((256, 32), Image.BILINEAR)

            try:
                text = vietocr_model.predict(roi_pil)
                recognized_texts.append(text)
            except:
                recognized_texts.append("[Lỗi OCR]")

    text_only = cv2.bitwise_and(image, image, mask=binary_mask)
    return text_only, recognized_texts

if __name__ == "__main__":
    image_path = r"E:\HocTap\KLTN\bien_hieu5.png"
    vietocr_model = load_vietocr_model()
    original, rotated = auto_rotate_image(image_path)

    if rotated is not None:
        text_image, ocr_texts = detect_text_with_vietocr(rotated, vietocr_model)

        num_lines = len(ocr_texts)
        fig_height = max(5, num_lines * 0.4)  # tự tăng chiều cao theo số dòng
        fig = plt.figure(figsize=(12, fig_height))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8])

        ax1 = plt.subplot(gs[0])
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax1.set_title("Ảnh gốc (đã xoay)")
        ax1.axis("off")

        ax2 = plt.subplot(gs[1])
        ax2.imshow(cv2.cvtColor(text_image, cv2.COLOR_BGR2RGB))
        ax2.set_title("Ảnh vùng chữ")
        ax2.axis("off")

        ax3 = plt.subplot(gs[2])
        ax3.axis("off")
        ax3.set_title("📝 Văn bản OCR", fontsize=12, loc='left')
        full_text = "\n".join([f"{i+1}. {txt}" for i, txt in enumerate(ocr_texts)])
        ax3.text(0, 1, full_text, fontsize=10, va='top', ha='left', wrap=True)

        plt.tight_layout()
        plt.show()
