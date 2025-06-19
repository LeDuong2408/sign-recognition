from io import BytesIO
import logging
from PIL import Image
import cv2
import httpx
import numpy as np

async def fetch_image(url: str) -> Image.Image:
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            logging.error(f'Failed to download image: {url}, error: {str(e)}')
            raise Exception(f"Failed to download image: {url}, error: {str(e)}")
