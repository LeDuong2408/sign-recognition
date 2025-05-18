# pip install -r requirements.txt &
pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 &
pip install numpy==1.24.3
conda install ninja -c conda-forge & 
cd OCR &
cd detector &
git clone https://github.com/aim-uofa/AdelaiDet.git &
cd AdelaiDet &
pip install -e . &
python -m pip install "git+https://github.com/facebookresearch/detectron2.git" &
pip install opencv-contrib-python==4.5.5.62 &
pip install opencv-python==4.6.0.66 &
pip install paddleocr==2.7.2 &
pip install paddlepaddle==2.6.2 &
pip install numpy==1.24.4 &
pip install vietocr &
pip install ultralytics==8.3.54 &
