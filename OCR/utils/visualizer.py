import numpy as np
import os
import cv2

def visualize_detection(img, points, filename, save_dir="output_vis"):
    """
    Visualize detection result on the image with bounding polygons and recognized text.

    Args:
        img (np.ndarray): Input image in BGR format.
        instances (detectron2.structures.Instances): Output from the predictor.
        filename (str): Filename of the image for saving.
        save_dir (str): Directory to save visualized output images.
    """
    os.makedirs(save_dir, exist_ok=True)
    img_vis = img.copy()

    for box in points:
        

        pts = np.array(box, int).reshape((-1, 1, 2))
        cv2.polylines(img_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    out_path = os.path.join(save_dir, f"v8-{filename}.jpg")
    cv2.imwrite(out_path, img_vis)
    print(f"[✔] Saved visualization to: {out_path}")