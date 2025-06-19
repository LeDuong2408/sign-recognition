import numpy as np
import os
import cv2


def visualize_detection_with_index(
    img, box_word, box_score, filename, save_dir="output_vis"
):
    """
    Visualize boxes with index number and optional score.

    Args:
        img (np.ndarray): Input image (BGR).
        box_word (List[List[List[float]]]): List of polygons, each with 4 points [[x1,y1],...].
        box_score (List[float]): List of scores, same length as box_word.
        filename (str): Image filename (without path).
        save_dir (str): Directory to save visualization.
    """
    os.makedirs(save_dir, exist_ok=True)
    img_vis = img.copy()

    for idx, (poly, score) in enumerate(zip(box_word, box_score)):
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))  # (4,1,2)

        # Draw polygon
        cv2.polylines(img_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Compute center
        center_x = int(np.mean([p[0] for p in poly]))
        center_y = int(np.mean([p[1] for p in poly]))

        # Draw index number
        cv2.putText(
            img_vis,
            str(idx + 1),
            (center_x, center_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 0, 255),
            thickness=2,
        )

    out_path = os.path.join(save_dir, f"indexed-{filename}.jpg")
    cv2.imwrite(out_path, img_vis)
    print(f"[✔] Saved indexed visualization to: {out_path}")


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


import os
import numpy as np
import cv2


def bezier_curve(points, num=200):
    """
    Tạo đường cong Bezier bậc 3 từ 4 điểm điều khiển.
    """
    points = np.array(points, dtype=np.float32)
    t = np.linspace(0, 1, num=num).reshape(-1, 1)
    curve = (
        (1 - t) ** 3 * points[0]
        + 3 * (1 - t) ** 2 * t * points[1]
        + 3 * (1 - t) * t**2 * points[2]
        + t**3 * points[3]
    )
    return curve


def visualize_detection_bezier(img, points_list, filename, save_dir="output_vis"):
    """
    Vẽ kết quả nhận diện (box Bezier 8 điểm) trên ảnh.

    Args:
        img (np.ndarray): Ảnh đầu vào (BGR).
        points_list (List[List[float]]): Danh sách các box (8 điểm: x1,y1,...,x8,y8).
        filename (str): Tên file ảnh (không có path).
        save_dir (str): Thư mục lưu ảnh đã vẽ.
    """
    os.makedirs(save_dir, exist_ok=True)
    img_vis = img.copy()

    for pts in points_list:
        if len(pts) != 8:
            continue  # Bỏ qua nếu không phải 8 điểm (x,y)

        # # Chuyển về (x, y) dạng [[x1,y1], ..., [x8,y8]]
        # pts = np.array(bezier_box, dtype=np.float32).reshape(8, 2)

        # Tách 2 đường cong: trên (0→3) và dưới (4→7)
        upper = bezier_curve(pts[0:4], num=50)
        lower = bezier_curve(pts[4:8][::-1], num=50)

        # Nối đường cong trên và dưới thành polygon kín
        polygon = np.concatenate([upper, lower[::-1]])

        polygon_int = polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            img_vis, [polygon_int], isClosed=True, color=(0, 255, 0), thickness=2
        )

    out_path = os.path.join(save_dir, f"v9-{filename}.jpg")
    cv2.imwrite(out_path, img_vis)
    print(f"[✔] Saved visualization to: {out_path}")
