import os
import shutil
import random
from tqdm import tqdm

img_dir = "../data/rec_train_data/img"
gt_file = "../data/rec_train_data/crop_gt.txt"

# Cấu hình đầu ra
output_dir = "../data"
train_img_dir = os.path.join(output_dir, "train_images")
val_img_dir = os.path.join(output_dir, "valid_images")
anno_train = os.path.join(output_dir, "annotation_train.txt")
anno_val = os.path.join(output_dir, "annotation_valid.txt")

# Tạo thư mục nếu chưa có
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)

# Tỉ lệ train/val
train_ratio = 0.9

# Đọc dữ liệu
with open(gt_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if '\t' in line]

# Xáo trộn và chia
random.shuffle(lines)
split_idx = int(len(lines) * train_ratio)
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

# Ghi annotation và copy ảnh
def write_and_copy(lines, anno_file, target_img_dir):
    with open(anno_file, "w", encoding="utf-8") as fw:
        for line in tqdm(lines):
            if not line or "\t" not in line:
                    continue
            try:
                img_path, label = line.split("\t")
                
                
            except:
                print(f"❌ Lỗi tách dòng: '{line}'")
                continue    
            img_name = os.path.basename(img_path)
            src_img = os.path.join(img_dir, img_name)
            dst_img = os.path.join(target_img_dir, img_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
                fw.write(f"{img_name}\t{label}\n")
            else:
                print(f"⚠️ Không tìm thấy ảnh: {src_img}")

write_and_copy(train_lines, anno_train, train_img_dir)
write_and_copy(val_lines, anno_val, val_img_dir)

print("✅ Hoàn tất tạo dữ liệu huấn luyện cho VietOCR!")
