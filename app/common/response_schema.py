#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Generic, TypeVar, Union

from fastapi import Response
from pydantic import BaseModel, Field

from app.common.response_code import CustomResponse, CustomResponseCode
from app.utils.serializers import MsgSpecJSONResponse

SchemaT = TypeVar('SchemaT')


class ResponseModel(BaseModel):
    """

        @router.get('/test', response_model=ResponseModel)
        def test():
            return ResponseModel(data={'test': 'test'})


        @router.get('/test')
        def test() -> ResponseModel:
            return ResponseModel(data={'test': 'test'})


        @router.get('/test')
        def test() -> ResponseModel:
            res = CustomResponseCode.HTTP_200
            return ResponseModel(code=res.code, msg=res.msg, data={'test': 'test'})
    """

    code: int = Field(CustomResponseCode.HTTP_200.code, description='Return status code')
    msg: str = Field(CustomResponseCode.HTTP_200.msg, description='Return message')
    data: Union[Any, None] = Field(None, description='Return data')


class ResponseSchemaModel(ResponseModel, Generic[SchemaT]):
    """

        @router.get('/test', response_model=ResponseSchemaModel[GetApiDetail])
        def test():
            return ResponseSchemaModel[GetApiDetail](data=GetApiDetail(...))


        @router.get('/test')
        def test() -> ResponseSchemaModel[GetApiDetail]:
            return ResponseSchemaModel[GetApiDetail](data=GetApiDetail(...))


        @router.get('/test')
        def test() -> ResponseSchemaModel[GetApiDetail]:
            res = CustomResponseCode.HTTP_200
            return ResponseSchemaModel[GetApiDetail](code=res.code, msg=res.msg, data=GetApiDetail(...))
    """

    data: SchemaT


class ResponseBase:
    @staticmethod
    def _make_response_data(
        res: Union[CustomResponseCode, CustomResponse] = CustomResponseCode.HTTP_200,
        data: Any = None
    ) -> dict:
        return {"code": res.code, "msg": res.msg, "data": data}

    def __call__(
        self,
        *,
        res: Union[CustomResponseCode, CustomResponse] = CustomResponseCode.HTTP_200,
        data: Any = None
    ) -> Response:
        return self.success(res=res, data=data)

    def success(
        self,
        *,
        res: Union[CustomResponseCode, CustomResponse] = CustomResponseCode.HTTP_200,
        data: Any = None
    ) -> Response:
        return MsgSpecJSONResponse(content=self._make_response_data(res=res, 
                            data=data.model_dump() if isinstance(data, BaseModel) else data))

    def fail(
        self,
        *,
        res: Union[CustomResponseCode, CustomResponse] = CustomResponseCode.HTTP_400,
        data: Any = None
    ) -> Response:
        return MsgSpecJSONResponse(content=self._make_response_data(res=res,
                                    data=data.model_dump() if isinstance(data, BaseModel) else data))

    @staticmethod
    def fast_success(
        *,
        res: Union[CustomResponseCode, CustomResponse] = CustomResponseCode.HTTP_200,
        data: Any = None
    ) -> Response:
        return MsgSpecJSONResponse(content={
            "code": res.code,
            "msg": res.msg,
            "data": data.model_dump() if isinstance(data, BaseModel) else data
        })

