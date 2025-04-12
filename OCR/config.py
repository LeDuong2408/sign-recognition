import torch 
PATH_VGG_MODEL = './OCR/experiments/vgg_transformer.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
LANG = 'vi'
PATH_IMAGE = "./OCR/data/bien_hieu5.png"