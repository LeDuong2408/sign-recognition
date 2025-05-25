docker pull vanwdaiii/kltn-cpu:v2
docker run -it -v F:/University_HCMUTE/N4_HK2/sign-recognition:/app -w /app -p 8000:8000 --name kltn vanwdaiii/kltn-cpu:v2 bash 
conda activate env
# cd OCR/detector/AdelaiDet/
# pip install -e .
# cd /app/
# apt-get update && apt-get install -y nano
# # find /opt/conda/envs/env/lib/ -name paddleocr.py
# nano /opt/conda/envs/env/lib/python3.9/site-packages/paddleocr/paddleocr.py
# # ( Ctrl + W paste : if not dt_boxes: => if dt_boxes is None: )
python -m OCR.main --input OCR/input_image/ --output OCR/output/ --confidence_threshold 0.1 --weights OCR/weights/abcnetv2.pth
