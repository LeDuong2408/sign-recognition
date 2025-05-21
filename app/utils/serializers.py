#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any

from msgspec import json
from starlette.responses import JSONResponse

class MsgSpecJSONResponse(JSONResponse):
    """
    MsgSpec JSON
    """
    def render(self, content: Any) -> bytes:
        return json.encode(content)
