# Sign recognition using YOLOv8, ABCNetv2, VietOCR, LLMs, ELECTRA-base Vietnamese
# Guide to run Project
After pull project from Github, step-by-step to run project
## 1. Download checkpoints
Download all model and move to direction `OCR/weights/` from [link](https://drive.google.com/drive/u/0/folders/1TinwA3Wb7tTehJG8SK-wwDX706e6EWLF)
## 2. Config variable
1. Access and get API KEY CLERK at [link](https://dashboard.clerk.com/apps/app_2xXfHXmKWi1ZKgn5mgjJw0TlTCK/instances/ins_2xXfHa6pIQK2NBOJIbqRtQ7JJhq/api-keys) 
2. Paste API Key to `/web/.env.local`
```env
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=...
CLERK_SECRET_KEY=...
```
3. Create file `/web/.env`:
```env
OCR_API_URL=http://localhost:8000/api/v1/inference-pipline-ocr
UPLOAD_PRESET=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRECT=...
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_URL=...
DATABASE_URL=...
STRIPE_SECRET_KEY=sk_test_dummy
STRIPE_WEBHOOK_SECRET=whsec_dummy
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_test_dummy
```
4. Access and get API KEY Gemini at [Link](https://aistudio.google.com/apikey)
PASTE API KEY to `.env` file:
```env
GEMINI_API_KEY=...
```
## 3. Install modules detector
```bash
cd OCR/detector
git clone https://github.com/aim-uofa/AdelaiDet.git
```
Add funtion to class Beziers in `./OCR/detector/AdelaiDet/adet/structures/beziers.py`:
```python
def __len__(self) -> int:
    """
    Returns:
        int: The number of bezier curves.
    """
    return self.tensor.size(0)
```
## 4. Prepare enviroment
### Pull docker image 
```bash
docker pull leduong2408/sign-recognition:v1
```
## 5. Run backend:
Turn on CMD and run:
```bash
docker-compose up --build
```
(Note: before run backend, check path-to-project in docker-compose.yml correctly)
## 6. Run frontend:
Turn on CMD and run:
```bash
cd web
npm install
npm run dev
```
