#!/bin/bash

# Màu cho terminal
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Nạp biến môi trường
source ~/.bashrc

# Kích hoạt conda env
conda activate myenv

echo -e "${GREEN}Starting FastAPI backend on http://0.0.0.0:8000...${NC}"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &

echo -e "${GREEN}Starting frontend (npm run dev) at http://localhost:3000...${NC}"
cd web && npm run dev

wait

# tail -f backend.log
# tail -f frontend.log

