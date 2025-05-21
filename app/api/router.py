import asyncio
from fastapi import FastAPI, APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel # type: ignore
from typing import Annotated
from fastapi import Depends, FastAPI


from app.common.response_schema import ResponseBase
from app.utils.process_img import fetch_image
from app.core.conf import settings
from pipline import pipline
router = APIRouter(prefix=settings.FASTAPI_API_V1_PATH)

class ImageURLs(BaseModel):
    urls: list[str]

class TextOcrResp(BaseModel):
    raw_text: list[str]
    corrected_text: list[str]

def get_response_base():
    return ResponseBase()

@router.get('/')
def home():
    return "Hello World!"
@router.post('/inference-pipline-ocr')
async def sign_board_recognition(img_urls: ImageURLs, RESPONSE: Annotated[ResponseBase, Depends(get_response_base)]):
    
    images = await asyncio.gather(*(fetch_image(url) for url in img_urls.urls))
    
    raw_text, corrected_text = pipline(imgs=images)
    resp = TextOcrResp(raw_text=raw_text, corrected_text=corrected_text)
    return RESPONSE(data=resp)