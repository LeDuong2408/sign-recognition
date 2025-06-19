import numpy as np
from sklearn.cluster import AgglomerativeClustering

from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from OCR.utils.image_utils import pad_bbox

# #TODO
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering


# --- Box feature utilities ---
def get_box_center(box):
    box = np.array(box)
    return np.mean(box[:, 0]), np.mean(box[:, 1])


def get_box_top(box):
    return np.min(np.array(box)[:, 1])


def get_box_left(box):
    return np.min(np.array(box)[:, 0])


def reorder_text(boxes_word, texts, layout_eps=100, line_thresh=35):
    if not boxes_word:
        return boxes_word, texts

    centers = np.array([get_box_center(box) for box in boxes_word])
    heights = [
        np.max(np.array(box)[:, 1]) - np.min(np.array(box)[:, 1]) for box in boxes_word
    ]

    # --- Step 1: Phân cụm layout block bằng DBSCAN ---
    layout_labels = DBSCAN(eps=layout_eps, min_samples=1).fit_predict(centers)

    final_boxes, final_texts = [], []
    for layout in np.unique(layout_labels):
        idxs = np.where(layout_labels == layout)[0]
        block_boxes = [boxes_word[i] for i in idxs]
        block_texts = [texts[i] for i in idxs]

        # --- Step 2: Phân cụm dòng trong layout ---
        block_centers = np.array([get_box_center(box) for box in block_boxes])
        y_coords = block_centers[:, 1].reshape(-1, 1)

        if len(block_boxes) > 1:
            line_labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=line_thresh
            ).fit_predict(y_coords)
        else:
            line_labels = np.zeros(1, dtype=int)

        # Nhóm box theo dòng
        lines = {}
        for i, label in enumerate(line_labels):
            lines.setdefault(label, []).append((block_boxes[i], block_texts[i]))

        # Sort dòng theo y-top, trong dòng sort trái qua phải
        line_items = []
        for items in lines.values():
            y_top = np.median([get_box_top(b[0]) for b in items])
            items_sorted = sorted(items, key=lambda x: get_box_left(x[0]))
            line_items.append((y_top, items_sorted))

        line_items.sort(key=lambda x: x[0])
        for _, items in line_items:
            for box, text in items:
                final_boxes.append(box)
                final_texts.append(text)

    return final_boxes, final_texts


# TODO
# import numpy as np
# from sklearn.cluster import DBSCAN, AgglomerativeClustering
# from sklearn.decomposition import PCA


# # --- Box feature utilities ---
# def get_box_center(box):
#     box = np.array(box)
#     return np.mean(box[:, 0]), np.mean(box[:, 1])


# def get_box_top(box):
#     return np.min(np.array(box)[:, 1])


# def get_box_left(box):
#     return np.min(np.array(box)[:, 0])


# def reorder_text(boxes_word, texts, layout_eps=100, line_thresh_ratio=0.8):
#     if not boxes_word:
#         return boxes_word, texts

#     # Bước 1: Phân cụm layout theo trung tâm box
#     centers = np.array([get_box_center(box) for box in boxes_word])
#     layout_labels = DBSCAN(eps=layout_eps, min_samples=1).fit_predict(centers)

#     final_boxes, final_texts = [], []

#     for layout in np.unique(layout_labels):
#         idxs = np.where(layout_labels == layout)[0]
#         block_boxes = [boxes_word[i] for i in idxs]
#         block_texts = [texts[i] for i in idxs]
#         block_centers = np.array([get_box_center(box) for box in block_boxes])

#         # Bước 2: Dùng PCA để tìm hướng dòng chữ
#         if len(block_boxes) > 1:
#             # PCA để tìm trục dòng
#             pca = PCA(n_components=2)
#             pca.fit(block_centers)
#             primary_axis = pca.components_[1]  # trục vuông góc dòng
#             projected = block_centers @ primary_axis.T
#             projected = projected.reshape(-1, 1)

#             # Phân cụm dòng theo project
#             line_labels = AgglomerativeClustering(
#                 n_clusters=None, distance_threshold=avg_height * line_thresh_ratio
#             ).fit_predict(projected)
#         else:
#             # chỉ 1 box → coi như 1 dòng
#             line_labels = np.zeros(1, dtype=int)

#         # Gom lại theo dòng
#         lines = {}
#         for i, label in enumerate(line_labels):
#             lines.setdefault(label, []).append((block_boxes[i], block_texts[i]))

#         # Sắp xếp dòng theo top y, trong dòng sort trái → phải
#         line_items = []
#         for items in lines.values():
#             y_top = np.median([get_box_top(b[0]) for b in items])
#             sorted_line = sorted(items, key=lambda x: get_box_left(x[0]))  # trái → phải
#             line_items.append((y_top, sorted_line))

#         # Sắp dòng từ trên xuống
#         line_items.sort(key=lambda x: x[0])

#         for _, items in line_items:
#             for box, text in items:
#                 final_boxes.append(box)
#                 final_texts.append(text)

#     return final_boxes, final_texts


# #TODO
# import numpy as np
# import cv2
# import layoutparser as lp
# from sklearn.cluster import AgglomerativeClustering


# # --- Box feature utilities ---
# def get_box_center(box):
#     box = np.array(box)
#     return np.mean(box[:, 0]), np.mean(box[:, 1])


# def get_box_top(box):
#     return np.min(np.array(box)[:, 1])


# def get_box_left(box):
#     return np.min(np.array(box)[:, 0])


# def reorder_text(image, boxes_word, texts, line_thresh=35):
#     if not boxes_word:
#         return boxes_word, texts

#     # Step 0: Phân tích layout bằng LayoutParser
#     model = lp.Detectron2LayoutModel(
#         r"C:\University-HCMUTE\KLTN\code\sign-recognition\OCR\config\layoutparse\faster_rcnn_R_50_FPN_3x.yaml",
#         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
#         label_map={0: "Text"},
#     )

#     layout = model.detect(image)

#     # Tạo danh sách các block layout để phân nhóm box OCR
#     block_regions = [b for b in layout if b.type == "Text"]
#     print("Block: ", block_regions)

#     # Gán mỗi box_word vào block gần nhất nếu nó nằm trong block đó
#     block_groups = [[] for _ in block_regions]
#     unassigned = []

#     for box, text in zip(boxes_word, texts):
#         cx, cy = get_box_center(box)
#         assigned = False
#         for i, region in enumerate(block_regions):
#             if (
#                 region.block.x_1 <= cx <= region.block.x_2
#                 and region.block.y_1 <= cy <= region.block.y_2
#             ):
#                 block_groups[i].append((box, text))
#                 assigned = True
#                 break
#         if not assigned:
#             unassigned.append((box, text))

#     # Gom kết quả từng block
#     final_boxes, final_texts = [], []

#     def process_block(block):
#         block_boxes = [b for b, _ in block]
#         block_texts = [t for _, t in block]
#         centers = np.array([get_box_center(box) for box in block_boxes])
#         y_coords = centers[:, 1].reshape(-1, 1)

#         if len(block_boxes) > 1:
#             line_labels = AgglomerativeClustering(
#                 n_clusters=None, distance_threshold=line_thresh
#             ).fit_predict(y_coords)
#         else:
#             line_labels = np.zeros(1, dtype=int)

#         lines = {}
#         for i, label in enumerate(line_labels):
#             lines.setdefault(label, []).append((block_boxes[i], block_texts[i]))

#         line_items = []
#         for items in lines.values():
#             y_top = np.median([get_box_top(b[0]) for b in items])
#             items_sorted = sorted(items, key=lambda x: get_box_left(x[0]))
#             line_items.append((y_top, items_sorted))

#         line_items.sort(key=lambda x: x[0])
#         for _, items in line_items:
#             for box, text in items:
#                 final_boxes.append(box)
#                 final_texts.append(text)

#     for block in block_groups:
#         if block:
#             process_block(block)

#     if unassigned:
#         process_block(unassigned)

#     layout_boxes = [
#         [
#             [r.block.x_1, r.block.y_1],
#             [r.block.x_2, r.block.y_1],
#             [r.block.x_2, r.block.y_2],
#             [r.block.x_1, r.block.y_2],
#         ]
#         for r in block_regions
#     ]

#     return final_boxes, final_texts, layout_boxes


# TODO
# import numpy as np
# from sklearn.cluster import AgglomerativeClustering


# # --- Box feature utilities ---
# def get_box_center(box):
#     box = np.array(box, dtype=np.float32)
#     return float(np.mean(box[:, 0])), float(np.mean(box[:, 1]))


# def get_box_left(box):
#     return float(np.min(np.array(box)[:, 0]))


# def reorder_text(
#     boxes_word, texts, lines_word_boxes, dynamic_thresh=False, base_thresh=35
# ):
#     if not boxes_word:
#         return boxes_word, texts

#     # Step 1: Cluster lines into blocks to avoid horizontal merging of distant columns
#     line_centers = np.array([get_box_center(b) for b in lines_word_boxes])
#     line_y = line_centers[:, 1].reshape(-1, 1)
#     block_labels = AgglomerativeClustering(
#         n_clusters=None, distance_threshold=base_thresh
#     ).fit_predict(line_y)

#     # Step 2: Assign words to closest line (center inside line box)
#     def is_word_in_line(word_box, line_box):
#         cx, cy = get_box_center(word_box)
#         x1 = min(pt[0] for pt in line_box)
#         x2 = max(pt[0] for pt in line_box)
#         y1 = min(pt[1] for pt in line_box)
#         y2 = max(pt[1] for pt in line_box)
#         return x1 <= cx <= x2 and y1 <= cy <= y2

#     line_to_words = {i: [] for i in range(len(lines_word_boxes))}
#     word_assigned = set()

#     for wi, wbox in enumerate(boxes_word):
#         for li, lbox in enumerate(lines_word_boxes):
#             if is_word_in_line(wbox, lbox):
#                 line_to_words[li].append((wbox, texts[wi]))
#                 word_assigned.add(wi)
#                 break  # only assign to first matching line

#     # Step 3: Sort words in each block by line-top then left-to-right
#     block_lines = {}
#     for li, label in enumerate(block_labels):
#         block_lines.setdefault(label, []).append((li, lines_word_boxes[li]))

#     final_boxes, final_texts = [], []
#     for _, lines in sorted(block_lines.items()):
#         line_items = []
#         for li, _ in sorted(lines, key=lambda x: get_box_center(x[1])[1]):
#             word_items = sorted(
#                 line_to_words.get(li, []), key=lambda x: get_box_left(x[0])
#             )
#             line_items.extend(word_items)
#         for box, text in line_items:
#             final_boxes.append(box)
#             final_texts.append(text)

#     # Step 4: Handle unassigned words
#     unassigned = [i for i in range(len(boxes_word)) if i not in word_assigned]
#     if unassigned:
#         ungrouped_boxes = [boxes_word[i] for i in unassigned]
#         ungrouped_texts = [texts[i] for i in unassigned]
#         centers = np.array([get_box_center(b) for b in ungrouped_boxes])
#         y_coords = centers[:, 1].reshape(-1, 1)
#         heights = [
#             np.max(np.array(b)[:, 1]) - np.min(np.array(b)[:, 1])
#             for b in ungrouped_boxes
#         ]
#         med_h = np.median(heights)
#         thresh = float(np.clip(1.2 * med_h, 15, 50)) if dynamic_thresh else base_thresh

#         if len(ungrouped_boxes) > 1:
#             line_labels = AgglomerativeClustering(
#                 n_clusters=None, distance_threshold=thresh
#             ).fit_predict(y_coords)
#         else:
#             line_labels = np.zeros(1, dtype=int)

#         lines = {}
#         for i, label in enumerate(line_labels):
#             lines.setdefault(label, []).append((ungrouped_boxes[i], ungrouped_texts[i]))

#         line_items = []
#         for items in lines.values():
#             y_key = np.median([get_box_center(b)[1] for b, _ in items])
#             items_sorted = sorted(items, key=lambda x: get_box_left(x[0]))
#             line_items.append((y_key, items_sorted))

#         line_items.sort(key=lambda x: x[0])
#         for _, items in line_items:
#             for box, text in items:
#                 final_boxes.append(box)
#                 final_texts.append(text)

#     return final_boxes, final_texts


# # TODO
# TODO =================================================================================================================


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
    new_lines = fallback_cluster_unmatched_words(
        unmatched_word_boxes, unmatched_texts, line_boxes, line_groups, shape
    )
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
    gaps = [center_ys[i + 1] - center_ys[i] for i in range(len(center_ys) - 1)]
    return np.mean(gaps)


def adaptive_eps(word_boxes):
    heights = [np.linalg.norm(box[0][1] - box[3][1]) for box in word_boxes]
    avg_height = np.mean(heights)
    std_height = np.std(heights)
    vertical_gaps = compute_vertical_gaps(word_boxes)

    eps_factor = 0.4 + min(0.6, std_height / avg_height + vertical_gaps / avg_height)

    return avg_height * eps_factor


def fallback_cluster_unmatched_words(
    unmatched_word_boxes,
    unmatched_texts,
    line_boxes,
    line_groups,
    shape,
    eps_ratio=0.7,
    alpha=0.15,
    iow_merge_thresh=0.15,
):
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
    # alpha = avg_height / width_range

    # X = np.array([
    #     [float(x * alpha), float(y)] for x, y in zip(centers_x, centers_y)
    # ])

    X = np.array(centers_y).reshape(-1, 1).astype(float)

    try:
        db = DBSCAN(eps=avg_height * eps_ratio, min_samples=1).fit(X)
    except:
        print(X)
        raise Exception
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
