# docker pull vanwdaiii/kltn:v1
# docker run -it --rm -v C:/University-HCMUTE/KLTN/code/sign-recognition:/app -w /app kltn
docker run -it --rm -v C:/University-HCMUTE/KLTN/code/sign-recognition:/app -w /app vanwdaiii/kltn:v1 bash


apt-get update && apt-get install -y nano
find /opt/conda/envs/env/lib/ -name paddleocr.py
nano /opt/conda/envs/env/lib/python3.9/site-packages/paddleocr/paddleocr.py
# ( Ctrl + W paste : if not dt_boxes: => if dt_boxes is None: )
