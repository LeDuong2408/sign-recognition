import asyncio
import json
import logging
import os
import re
from typing import Literal
from google import genai
from google.genai import types
from POST_OCR.utils.preprocess_output_gemini import decode_unicode_if_needed
from app.core.conf import settings


class OCRCorrection:
    def __init__(
        self,
        model_name: Literal[
            "gemini-2.0-flash", "gemini-2.5-flash-preview-04-17"
        ] = "gemini-2.5-flash-preview-04-17",
        temperature: float = 0.4,
        response_mime_type: Literal["text/plain", "application/json"] = "text/plain",
        eval: bool = False,
    ):
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        self.client = genai.Client(
            api_key=api_key,
        )
        self.model_name = model_name
        # if not eval:
        self.content = self._get_content()
        self.sys_instruction = self._get_system_instruction()
        # else:
        #     self.content = self._get_content_eval()
        #     self.sys_instruction = self._get_system_instruction_eval()

        if model_name.startswith("gemini-2.5-flash"):
            thinking_config = types.ThinkingConfig(
                thinking_budget=1,
            )
        else:
            thinking_config = None
        self.config = types.GenerateContentConfig(
            thinking_config=thinking_config,
            temperature=temperature,
            response_mime_type=response_mime_type,
            system_instruction=self.sys_instruction,
        )

    def _get_system_instruction(self) -> list[types.Part]:
        system_instruction = """
              Bạn là một chuyên gia ngôn ngữ. 
              Nhiệm vụ của bạn là hiệu chỉnh văn bản được trích xuất bằng OCR từ các bảng hiệu quảng cáo. 
              Văn bản OCR có thể chứa lỗi như: nhận diện sai ký tự, thiếu ký tự, hoặc thêm ký tự không tồn tại trong văn bản gốc.
              Yêu cầu: 
              1 - Chỉ sửa lỗi dựa trên hiểu biết ngôn ngữ tự nhiên — không tự suy diễn hoặc bịa đặt nếu thông tin không chắc chắn.
              2 - Không đưa ra bất kỳ lời giải thích nào về nhiệm vụ hoặc kết quả trong phản hồi. 
              3 - Nếu văn bản chứa số điện thoại, không chèn dấu chấm giữa các cụm số — giữ nguyên chuỗi số liền mạch.
              4 - Chỉ trả về văn bản đã sửa lỗi, được bao quanh bởi thẻ <corrected> và </corrected> như trong HTML.
              """

        return [types.Part.from_text(text=system_instruction)]

    def _get_content(self) -> list[types.Content]:
        examples = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="""[\"TUBORG VI BIA TƯVÊY VỊ BIA TUYỆT HẢO TỪ 1880 TẠP HÓA ĐIỆN DÂN DỤNG  QUANG VINH CHUYÊN:  TẠP HOÁ TỔNG HỢP ĐẠI LÝ BIA,  NƯỚC NGỌT CÁC LOẠI 550, TÂY LẠC, AN CHU, BẮC SƠN,  TRẢNG BOM, ĐÔNG NAI\"]"""
                    ),
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="""[\"<corrected>TUBORG VỊ BIA TUYỆT HẢO TỪ 1880 TẠP HÓA ĐIỆN DÂN DỤNG QUANG VINH CHUYÊN: TẠP HOÁ TỔNG HỢP ĐẠI LÝ BIA, NƯỚC NGỌT CÁC LOẠI 550, TÂY LẠC, AN CHU, BẮC SƠN, TRẢNG BOM, ĐỒNG NAI</corrected>\"]"""
                    ),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="""[
						\"TẠP HÓA TƯỜNG OANH CocuCola NGHỈ XẢ HỢI ĐC:292Đường Đường Ga Tổ 12 - P. Quang Trung TP. Thái Nguyên HỮNG HỨNG KHỞI ĐT: 0374.023.805\", 
						\"TẬP HOÁ BIA VIỆT NGỌC TÂN GỐI KHÔNG bước UÔNG người DƯỚI 123 BIA RƯỢU - BÁNH KẸO - THUỐC LÁ NƯỚC GIẢI KHÁT CÁC LOẠI Đ/C: 140 ĐƯỜNG MINH CẦU - TP. THÁI NGUYÊN - Đ/T: 0989909393\"]"""
                    ),
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="""[
						\"<corrected>TẠP HÓA TƯỜNG OANH CocaCola NGHỈ XẢ HƠI ĐC:292 Đường Ga Tổ 12 - P. Quang Trung TP. Thái Nguyên BỪNG HỨNG KHỞI ĐT: 0374.023.805</corrected>\",
						\"<corrected>TẠP HÓA NGỌC TÂN BÁN: BIA - RƯỢU - BÁNH KẸO - THUỐC LÁ NƯỚC GIẢI KHÁT CÁC LOẠI (KHÔNG BÁN CHO NGƯỜI DƯỚI 18 TUỔI) Đ/C: 140 ĐƯỜNG MINH CẦU - TP. THÁI NGUYÊN - Đ/T: 0989909393</corrected>\"
						]"""
                    ),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="""[
						\"TẠP HÓA ĐIỆN DÂN DỤNG QUANG VINH CHUYÊN: TẠP HÓA TỔNG HỢP ĐẠI LÝ BIA, NƯỚC NGỌT CÁC LOẠI ĐT: ĐT:0973918466 550, TÂY LẠC, AN CHU, BẮC SƠN, TRẢNG 80M, ĐÔNG NAI\"]
						"""
                    ),
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="""[
						\"<corrected>TẠP HÓA ĐIỆN DÂN DỤNG QUANG VINH CHUYÊN: TẠP HÓA TỔNG HỢP ĐẠI LÝ BIA, NƯỚC NGỌT CÁC LOẠI ĐT: 0973918466 550, TÂY LẠC, AN CHU, BẮC SƠN, TRẢNG BOM, ĐỒNG NAI</corrected>\"]"""
                    ),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="""[
						\"CÔNG TY CỔ PHẦN CHĂN NUÔI C.P. VIỆT NAM Cửa hàng: CHẢO AN C.P. GROÚP Chuyên cung cầp: THỨC ĂN CHĂN LƯỚI CON GIÓNG : TINH HEO, THIẾT BỊ - DỤNG CỤ CHĂN NU... : DỊCH VU KETHUẬT. CP PHÁT TRIỂN CHĂN NUÕI BẾN VƯNG ĐC: ẤP 2, THUẬN HÒA, LONG MỸ, HẬU GIANG 01666.098.0953 0939.053502\",
						\"BIA ĐẠI LÝ LARUE SÁNH CHUYÊN LOAN SPECIAL CUNG CẤP SẼ SALE BIA NƯỚC GIÀI KHÁT SC ĐT: ẤP 18, : 92 THUẬN HƯNG, TE LONG MỸ, NỮ GIANG 67 0962671811\"]"""
                    ),
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="""[
						\"<corrected>CÔNG TY CỔ PHẦN CHĂN NUÔI C.P. VIỆT NAM Cửa hàng: CHẢO AN C.P. GROUP Chuyên cung cấp: THỨC ĂN CHĂN NUÔI CON GIỐNG: HEO, THIẾT BỊ - DỤNG CỤ CHĂN NUÔI DỊCH VỤ KỸ THUẬT. CP PHÁT TRIỂN CHĂN NUÔI BẾN VỮNG ĐC: ẤP 2, THUẬN HÒA, LONG MỸ, HẬU GIANG 09666.098.0953 - 0939.053502</corrected>\",
						\"<corrected>ĐẠI LÝ BIA LARUE SÀNH CHUYÊN LOAN SPECIAL CUNG CẤP SỈ VÀ LẺ BIA NƯỚC GIẢI KHÁT ĐT: ẤP 18, THUẬN HƯNG, LONG MỸ, HẬU GIANG 0962671811</corrected>\"
						]"""
                    ),
                ],
            ),
        ]

        return examples

    def _get_content_eval(self) -> list[types.Content]:
        examples = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="""['CẦN', 'PHỐI', 'www.mic.vn', 'KHÔNG', 'ĐƯỜNG', 'TẮC', 'thứ', 'ĐỜI', 'TRƯỜNG', '3%', 'SẼ', 'TIỆC', 'CHO', 'Song', 'MỤC', 'HỌC', 'Café', 'DĐ:', 'VỤ', '250ml)', 'HỒNG', 'Acnes', 'THUỐC', 'MỸ', 'Ơ', 'Sách', '57', 'SHOP', 'PHÔNG', 'đi', 'ĐẶNG', 'AN', 'nói', '100', 'PHỐ', '5', 'LONG', 'PHONG', 'SACH', 'MÃI', 'rồi', '4,2m', 'TP.TDM.BD', 'SẢN', 'KHAI', '1/13', 'tháng', 'DU', 'http:/haphaco.com', '-']"""
                    ),
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="""['<corrected>CẦN</corrected>', '<corrected>PHỐI</corrected>', '<corrected>www.mic.vn</corrected>', '<corrected>KHÔNG</corrected>', '<corrected>ĐƯỜNG</corrected>', '<corrected>TẮC</corrected>', '<corrected>thứ</corrected>', '<corrected>ĐỜI</corrected>', '<corrected>TRƯỜNG</corrected>', '<corrected>3%</corrected>', '<corrected>SẼ</corrected>', '<corrected>TIỆC</corrected>', '<corrected>CHO</corrected>', '<corrected>Song</corrected>', '<corrected>MỤC</corrected>', '<corrected>R</corrected>', '<corrected>Cafe</corrected>', '<corrected>DĐ:</corrected>', '<corrected>vụ</corrected>', '<corrected>250ml)</corrected>', '<corrected>HỒNG</corrected>', '<corrected>Acnes</corrected>', '<corrected>R</corrected>', '<corrected>MỸ</corrected>', '<corrected>Ơ</corrected>', '<corrected>Sách</corrected>', '<corrected>57</corrected>', '<corrected>SHOP</corrected>', '<corrected>PHÔNG</corrected>', '<corrected>đi</corrected>', '<corrected>ĐẶNG</corrected>', '<corrected>AN</corrected>', '<corrected>nói</corrected>', '<corrected>100</corrected>', '<corrected>PHỐ</corrected>', '<corrected>5</corrected>', '<corrected>LONG</corrected>', '<corrected>PHONG</corrected>', '<corrected>SACH</corrected>', '<corrected>MÃI</corrected>', '<corrected>rồi</corrected>', '<corrected>4,2m</corrected>', '<corrected>TP. TDM.BD</corrected>', '<corrected>SẢN</corrected>', '<corrected>KHAI</corrected>', '<corrected>1/13</corrected>', '<corrected>HÀNH</corrected>', '<corrected>DU</corrected>', '<corrected>http://haphaco.com</corrected>', '<corrected>-</corrected>']"""
                    ),
                ],
            ),
        ]
        return examples

    def _get_system_instruction_eval(self) -> list[types.Part]:
        system_instruction = """Bạn là một chuyên gia ngôn ngữ được giao nhiệm vụ sửa lỗi văn bản OCR ở word-level. \
              Văn bản OCR có thể chứa lỗi sai ký tự, bỏ sót ký tự hoặc thêm \
              ký tự không có trong văn bản gốc. Hãy sử dụng kiến thức ngôn ngữ của bạn để phát hiện và sửa các lỗi này. \ 
              Lưu ý: \
              1 - Không được bịa đặt thông tin bạn không biết để sửa lỗi. \
              2 - Trong phản hồi của bạn, không đưa ra giải thích về nhiệm vụ hoặc kết quả. \
              3 - Chỉ cung cấp văn bản được sửa lỗi. Hãy đưa câu trả lời của bạn vào giữa thẻ <corrected> và </corrected> như trong mã HTML."""

        return [types.Part.from_text(text=system_instruction)]

    def _to_type_content(self, content: list[str]) -> types.Content:
        return types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=json.dumps(content)),
            ],
        )

    def run(self, text: list[str]) -> list[str]:
        text = self._to_type_content(text)

        _content = self.content.copy()
        _content.append(text)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=_content,
            config=self.config,
        )
        logging.info(f"Response: {response.text}")
        result = self._extract_respone_from_tag(response.text)
        return result

    def __call__(self, text: list[str]) -> list[str]:
        return self.run(text)

    def _extract_respone_from_tag(self, response: str) -> list[str]:
        """
        Extract corrected text from respone Gemini
        eg: <corrected>[CORRECTED TEX]</corrected>
        """
        response = decode_unicode_if_needed(response)
        pattern = r"<corrected>(.*?)</corrected>"
        result = re.findall(pattern, response, re.DOTALL)

        return result


if __name__ == "__main__":
    input = ["abc da"]
    corrector = OCRCorrection()
    asyncio.run(corrector(input))
