#!/bin/bash

# Màu cho terminal
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Nạp biến môi trường
source ~/.bashrc

# Kích hoạt conda env
conda activate myenv

uvicorn app.main:app --port 8000 --log-level trace

wait