python -m OCR.main --input OCR/data/ --output OCR/output/ --confidence_threshold 0.1 --weights OCR/weights/abcnetv2.pth
python -m OCR.main --input OCR/input_image/ --output OCR/output/ --confidence_threshold 0.1 --weights OCR/weights/abcnetv2.pth
python -m OCR.main --input OCR/input_image/ --output OCR/output/
python -m OCR.main --input OCR/data/ --output OCR/output/
python -m OCR.main --input OCR/eval/data/imgs_eval_ocr --output OCR/eval/data/raw_labels_eval_ocr
python -m OCR.main --input OCR/eval/data/imgs_eval_ocr --output OCR/output/img_eval_ocr