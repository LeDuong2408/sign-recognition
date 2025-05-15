import torch 
PATH_VGG_MODEL = './OCR/experiments/vgg_transformer.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
LANG = 'vi'

PATH_IMAGE = "./OCR/data/8.jpg"
PATH_REALESRGAN_MODEL = './OCR/experiments/RealESRGAN_x4plus.pth'

