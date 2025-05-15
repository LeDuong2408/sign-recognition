# pip install -r requirements.txt &
conda install ninja -c conda-forge & 
python -m pip install "git+https://github.com/facebookresearch/detectron2.git" &
cd detector &
git clone https://github.com/aim-uofa/AdelaiDet.git &
cd AdelaiDet &
python setup.py build develop 