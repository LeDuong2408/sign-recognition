FROM continuumio/miniconda3

# Tạo môi trường Conda
RUN apt-get update && apt-get install -y build-essential gcc git

# Cài đặt gói bên trong shell đã kích hoạt conda env
RUN bash -c "\
    source /opt/conda/etc/profile.d/conda.sh && \
    conda create -n env python=3.9.21 -y && \
    conda activate env && \
    conda install ninja -c conda-forge -y && \
    mkdir -p /app/package && \
    git clone https://github.com/aim-uofa/AdelaiDet.git /app/package/AdelaiDet && \
    pip install numpy==1.24.4 && \
    pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 && \
    cd /app/package/AdelaiDet && pip install -e . && \
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' && \
    pip install opencv-contrib-python==4.5.5.62 && \
    pip install opencv-python==4.6.0.66 && \
    pip install paddleocr==2.7.2 && \
    pip install paddlepaddle==2.6.2 && \
    pip install vietocr && \
    pip install ultralytics==8.3.54 \
"

# Set thư mục làm việc
WORKDIR /app

# Thiết lập entrypoint vào conda env
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "env", "bash", "-c" ]
CMD ["bash"]
