import numpy as np

from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from OCR.utils.image_utils import pad_bbox

def box_center_y(box):
    return np.mean([point[1] for point in box])

def box_center_x(box):
    return np.mean([point[0] for point in box])

def box_center(box):
    x = np.mean([p[0] for p in box])
    y = np.mean([p[1] for p in box])
    return (x, y)

def compute_iou(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    if not poly1.is_valid or not poly2.is_valid:
        return 0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if union == 0:
        return 0
    return inter / union
def compute_iow(word_box, line_box):
    poly_word = Polygon(word_box)
    poly_line = Polygon(line_box)

    if not poly_word.is_valid or not poly_line.is_valid:
        return 0

    inter_area = poly_word.intersection(poly_line).area
    word_area = poly_word.area

    if word_area == 0:
        return 0

    return inter_area / word_area

def merge_boxes(boxes):
    all_x = [pt[0] for box in boxes for pt in box]
    all_y = [pt[1] for box in boxes for pt in box]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

def assign_words_to_lines(word_boxes, line_boxes, texts, shape, iow_thresh=0.45):
    line_groups = [[] for _ in line_boxes]
    unmatched_word_boxes = []
    unmatched_texts = []    
    for word_box, text in zip(word_boxes, texts):
        best_iou = 0
        best_line_idx = -1
        
        for idx, line_box in enumerate(line_boxes):
            iou = compute_iow(word_box, line_box)
            if iou > best_iou:
                best_iou = iou
                best_line_idx = idx
        if best_iou >= iow_thresh:
            line_groups[best_line_idx].append((word_box, text))
        else:
            unmatched_word_boxes.append(word_box)
            unmatched_texts.append(text)
    new_lines = fallback_cluster_unmatched_words(unmatched_word_boxes, unmatched_texts, line_boxes, line_groups, shape)
    for new_line in new_lines:
        line_groups.append(new_line)
        
    for group in line_groups:
        group.sort(key=lambda x: box_center_y(x[0]))
        group.sort(key=lambda x: box_center_x(x[0]))
        
    return line_groups

def compute_vertical_gaps(boxes):
    if len(boxes) < 2:
        return 0
    center_ys = [box_center_y(box) for box in boxes]
    center_ys.sort()
    gaps = [center_ys[i+1] - center_ys[i] for i in range(len(center_ys) - 1)]
    return np.mean(gaps)

def adaptive_eps(word_boxes):
    heights = [np.linalg.norm(box[0][1] - box[3][1]) for box in word_boxes]
    avg_height = np.mean(heights)          
    std_height = np.std(heights)      
    vertical_gaps = compute_vertical_gaps(word_boxes)  

    eps_factor = 0.4 + min(0.6, std_height / avg_height + vertical_gaps / avg_height)

    return avg_height * eps_factor

def fallback_cluster_unmatched_words(unmatched_word_boxes, unmatched_texts, line_boxes, line_groups, shape, eps_ratio=0.7, alpha=0.15, iow_merge_thresh=0.15):
    """
    Gom nhóm các từ không khớp với line nào thành các dòng mới bằng clustering,
    sử dụng (center_x * alpha, center_y) để giảm sai khi các từ cách xa nhau theo chiều ngang.
    """
    if not unmatched_word_boxes:
        return []

    centers_x = [box_center_x(box) for box in unmatched_word_boxes]
    centers_y = [box_center_y(box) for box in unmatched_word_boxes]
    
    eps = adaptive_eps(unmatched_word_boxes)
    
    heights = [np.linalg.norm(box[0][1] - box[3][1]) for box in unmatched_word_boxes]
    avg_height = np.mean(heights)
    
    width_range = np.max(centers_x) - np.min(centers_x)
    # Alpha là tỷ lệ giữa chiều cao trung bình và khoảng cách chiều ngang giảm phụ thuộc vào x
    alpha = avg_height / width_range
    
    X = np.array([
        [x * alpha, y] for x, y in zip(centers_x, centers_y)
    ])

    
    db = DBSCAN(eps=avg_height * eps_ratio, min_samples=1).fit(X)
    labels = db.labels_

    cluster_map = {}
    for label, box, text in zip(labels, unmatched_word_boxes, unmatched_texts):
        cluster_map.setdefault(label, []).append((box, text))
    
    new_lines = []

    for group in cluster_map.values():
        # tạo box bao quanh tất cả các từ trong nhóm
        # kiểm tra nó giao với line box nào thì gán vào đó
        # nếu không thì tạo một line mới
        # cluster_box = merge_boxes([box for box, _ in group])
        # cluster_box = pad_bbox(cluster_box, 10, shape) 
        # best_iow = 0
        # best_line_idx = -1
        # for idx, line_box in enumerate(line_boxes):
        #     iow = compute_iow(cluster_box, line_box)
        #     if iow > best_iow:
        #         best_iow = iow
        #         best_line_idx = idx
                
        # if best_iow >= iow_merge_thresh and best_line_idx != -1:
        #     line_groups[best_line_idx].extend(group)
        # else:
        #     new_lines.append(group)
        new_lines.append(group)
    return new_lines
