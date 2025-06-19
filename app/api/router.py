import asyncio
import logging
from fastapi import FastAPI, APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel  # type: ignore
from typing import Annotated
from fastapi import Depends, FastAPI


from app.common.response_schema import ResponseBase
from app.utils.process_img import fetch_image
from app.core.conf import settings
from pipline import pipline

router = APIRouter(prefix=settings.FASTAPI_API_V1_PATH)


class ImageURLs(BaseModel):
    img_urls: list[str]


class TextOcrResp(BaseModel):
    raw_text: list[str]
    corrected_text: list[str]
    ner_text: list[dict]


def get_response_base():
    return ResponseBase()


@router.get("/")
def home():
    return "Hello World!"


@router.post("/inference-pipline-ocr")
async def sign_board_recognition(
    body: ImageURLs, RESPONSE: Annotated[ResponseBase, Depends(get_response_base)]
):
    print("Start Load Images...")
    images = await asyncio.gather(*(fetch_image(url) for url in body.img_urls))
    print("Done Load Images")

    raw_text, corrected_text, ner_text = await pipline(imgs=images)
    return RESPONSE(data=ner_text)
