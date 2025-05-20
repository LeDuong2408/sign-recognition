import json
import os
import re
from google import genai
from google.genai import types

class OCRCorrection:
    def __init__(self, model_name: str, ):
        api_key=os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(
            api_key=api_key,
        )
        self.content = self.get_examples()
        
    def get_examples(self) -> list[types.Content]:
        examples = [
			types.Content(
				role="user",
				parts=[
					types.Part.from_text(text="""[\"TUBORG VI BIA TƯVÊY VỊ BIA TUYỆT HẢO TỪ 1880 TẠP HÓA ĐIỆN DÂN DỤNG  QUANG VINH CHUYÊN:  TẠP HOÁ TỔNG HỢP ĐẠI LÝ BIA,  NƯỚC NGỌT CÁC LOẠI 550, TÂY LẠC, AN CHU, BẮC SƠN,  TRẢNG BOM, ĐÔNG NAI\"]"""),
				],
			),
			types.Content(
				role="model",
				parts=[
					types.Part.from_text(text="""[\"<corrected>TUBORG VỊ BIA TUYỆT HẢO TỪ 1880 TẠP HÓA ĐIỆN DÂN DỤNG QUANG VINH CHUYÊN: TẠP HOÁ TỔNG HỢP ĐẠI LÝ BIA, NƯỚC NGỌT CÁC LOẠI 550, TÂY LẠC, AN CHU, BẮC SƠN, TRẢNG BOM, ĐỒNG NAI</corrected>\"]"""),
				],
			),
			types.Content(
				role="user",
				parts=[
					types.Part.from_text(text="""[
						\"TẠP HÓA TƯỜNG OANH CocuCola NGHỈ XẢ HỢI ĐC:292Đường Đường Ga Tổ 12 - P. Quang Trung TP. Thái Nguyên HỮNG HỨNG KHỞI ĐT: 0374.023.805\", 
						\"TẬP HOÁ BIA VIỆT NGỌC TÂN GỐI KHÔNG bước UÔNG người DƯỚI 123 BIA RƯỢU - BÁNH KẸO - THUỐC LÁ NƯỚC GIẢI KHÁT CÁC LOẠI Đ/C: 140 ĐƯỜNG MINH CẦU - TP. THÁI NGUYÊN - Đ/T: 0989909393\"]"""),
				],
			),
			types.Content(
				role="model",
				parts=[
					types.Part.from_text(text="""[
						\"<corrected>TẠP HÓA TƯỜNG OANH CocaCola NGHỈ XẢ HƠI ĐC:292 Đường Ga Tổ 12 - P. Quang Trung TP. Thái Nguyên BỪNG HỨNG KHỞI ĐT: 0374.023.805</corrected>\",
						\"<corrected>TẠP HÓA NGỌC TÂN BÁN: BIA - RƯỢU - BÁNH KẸO - THUỐC LÁ NƯỚC GIẢI KHÁT CÁC LOẠI (KHÔNG BÁN CHO NGƯỜI DƯỚI 18 TUỔI) Đ/C: 140 ĐƯỜNG MINH CẦU - TP. THÁI NGUYÊN - Đ/T: 0989909393</corrected>\"
						]"""),
				],
			),
			types.Content(
				role="user",
				parts=[
					types.Part.from_text(text="""[
						\"TẠP HÓA ĐIỆN DÂN DỤNG QUANG VINH CHUYÊN: TẠP HÓA TỔNG HỢP ĐẠI LÝ BIA, NƯỚC NGỌT CÁC LOẠI ĐT: ĐT:0973918466 550, TÂY LẠC, AN CHU, BẮC SƠN, TRẢNG 80M, ĐÔNG NAI\"]
						"""),
				],
			),
			types.Content(
				role="model",
				parts=[
					types.Part.from_text(text="""[
						\"<corrected>TẠP HÓA ĐIỆN DÂN DỤNG QUANG VINH CHUYÊN: TẠP HÓA TỔNG HỢP ĐẠI LÝ BIA, NƯỚC NGỌT CÁC LOẠI ĐT: 0973918466 550, TÂY LẠC, AN CHU, BẮC SƠN, TRẢNG BOM, ĐỒNG NAI</corrected>\"]"""),
				],
			),
			types.Content(
				role="user",
				parts=[
					types.Part.from_text(text="""[
						\"CÔNG TY CỔ PHẦN CHĂN NUÔI C.P. VIỆT NAM Cửa hàng: CHẢO AN C.P. GROÚP Chuyên cung cầp: THỨC ĂN CHĂN LƯỚI CON GIÓNG : TINH HEO, THIẾT BỊ - DỤNG CỤ CHĂN NU... : DỊCH VU KETHUẬT. CP PHÁT TRIỂN CHĂN NUÕI BẾN VƯNG ĐC: ẤP 2, THUẬN HÒA, LONG MỸ, HẬU GIANG 01666.098.0953 0939.053502\",
						\"BIA ĐẠI LÝ LARUE SÁNH CHUYÊN LOAN SPECIAL CUNG CẤP SẼ SALE BIA NƯỚC GIÀI KHÁT SC ĐT: ẤP 18, : 92 THUẬN HƯNG, TE LONG MỸ, NỮ GIANG 67 0962671811\"]"""),
				],
			),
			types.Content(
				role="model",
				parts=[
					types.Part.from_text(text="""[
						\"<corrected>CÔNG TY CỔ PHẦN CHĂN NUÔI C.P. VIỆT NAM Cửa hàng: CHẢO AN C.P. GROUP Chuyên cung cấp: THỨC ĂN CHĂN NUÔI CON GIỐNG: HEO, THIẾT BỊ - DỤNG CỤ CHĂN NUÔI DỊCH VỤ KỸ THUẬT. CP PHÁT TRIỂN CHĂN NUÔI BẾN VỮNG ĐC: ẤP 2, THUẬN HÒA, LONG MỸ, HẬU GIANG 09666.098.0953 - 0939.053502</corrected>\",
						\"<corrected>ĐẠI LÝ BIA LARUE SÀNH CHUYÊN LOAN SPECIAL CUNG CẤP SỈ VÀ LẺ BIA NƯỚC GIẢI KHÁT ĐT: ẤP 18, THUẬN HƯNG, LONG MỸ, HẬU GIANG 0962671811</corrected>\"
						]"""),
				],
			),
		]
        
        return examples
        
        


def generate(input: list[str]) -> list[str]:
    GEMINI_API_KEY = 'AIzaSyAT59AiN0NCWnNkJz5dJiknD9Ghwa7plpE'
    # client = genai.Client(
    #     api_key=os.environ.get("GEMINI_API_KEY"),
    # )
    client = genai.Client(
        api_key=GEMINI_API_KEY,
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""[\"TUBORG VI BIA TƯVÊY VỊ BIA TUYỆT HẢO TỪ 1880 TẠP HÓA ĐIỆN DÂN DỤNG  QUANG VINH CHUYÊN:  TẠP HOÁ TỔNG HỢP ĐẠI LÝ BIA,  NƯỚC NGỌT CÁC LOẠI 550, TÂY LẠC, AN CHU, BẮC SƠN,  TRẢNG BOM, ĐÔNG NAI\"]"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""[\"<corrected>TUBORG VỊ BIA TUYỆT HẢO TỪ 1880 TẠP HÓA ĐIỆN DÂN DỤNG QUANG VINH CHUYÊN: TẠP HOÁ TỔNG HỢP ĐẠI LÝ BIA, NƯỚC NGỌT CÁC LOẠI 550, TÂY LẠC, AN CHU, BẮC SƠN, TRẢNG BOM, ĐỒNG NAI</corrected>\"]"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""[
    \"TẠP HÓA TƯỜNG OANH CocuCola NGHỈ XẢ HỢI ĐC:292Đường Đường Ga Tổ 12 - P. Quang Trung TP. Thái Nguyên HỮNG HỨNG KHỞI ĐT: 0374.023.805\", 
    \"TẬP HOÁ BIA VIỆT NGỌC TÂN GỐI KHÔNG bước UÔNG người DƯỚI 123 BIA RƯỢU - BÁNH KẸO - THUỐC LÁ NƯỚC GIẢI KHÁT CÁC LOẠI Đ/C: 140 ĐƯỜNG MINH CẦU - TP. THÁI NGUYÊN - Đ/T: 0989909393\"]"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""[
  \"<corrected>TẠP HÓA TƯỜNG OANH CocaCola NGHỈ XẢ HƠI ĐC:292 Đường Ga Tổ 12 - P. Quang Trung TP. Thái Nguyên BỪNG HỨNG KHỞI ĐT: 0374.023.805</corrected>\",
  \"<corrected>TẠP HÓA NGỌC TÂN BÁN: BIA - RƯỢU - BÁNH KẸO - THUỐC LÁ NƯỚC GIẢI KHÁT CÁC LOẠI (KHÔNG BÁN CHO NGƯỜI DƯỚI 18 TUỔI) Đ/C: 140 ĐƯỜNG MINH CẦU - TP. THÁI NGUYÊN - Đ/T: 0989909393</corrected>\"
]"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""[
  \"TẠP HÓA ĐIỆN DÂN DỤNG QUANG VINH CHUYÊN: TẠP HÓA TỔNG HỢP ĐẠI LÝ BIA, NƯỚC NGỌT CÁC LOẠI ĐT: ĐT:0973918466 550, TÂY LẠC, AN CHU, BẮC SƠN, TRẢNG 80M, ĐÔNG NAI\"]
"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""[
  \"<corrected>TẠP HÓA ĐIỆN DÂN DỤNG QUANG VINH CHUYÊN: TẠP HÓA TỔNG HỢP ĐẠI LÝ BIA, NƯỚC NGỌT CÁC LOẠI ĐT: 0973918466 550, TÂY LẠC, AN CHU, BẮC SƠN, TRẢNG BOM, ĐỒNG NAI</corrected>\"]"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""[
  \"CÔNG TY CỔ PHẦN CHĂN NUÔI C.P. VIỆT NAM Cửa hàng: CHẢO AN C.P. GROÚP Chuyên cung cầp: THỨC ĂN CHĂN LƯỚI CON GIÓNG : TINH HEO, THIẾT BỊ - DỤNG CỤ CHĂN NU... : DỊCH VU KETHUẬT. CP PHÁT TRIỂN CHĂN NUÕI BẾN VƯNG ĐC: ẤP 2, THUẬN HÒA, LONG MỸ, HẬU GIANG 01666.098.0953 0939.053502\",
  \"BIA ĐẠI LÝ LARUE SÁNH CHUYÊN LOAN SPECIAL CUNG CẤP SẼ SALE BIA NƯỚC GIÀI KHÁT SC ĐT: ẤP 18, : 92 THUẬN HƯNG, TE LONG MỸ, NỮ GIANG 67 0962671811\"]"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""[
  \"<corrected>CÔNG TY CỔ PHẦN CHĂN NUÔI C.P. VIỆT NAM Cửa hàng: CHẢO AN C.P. GROUP Chuyên cung cấp: THỨC ĂN CHĂN NUÔI CON GIỐNG: HEO, THIẾT BỊ - DỤNG CỤ CHĂN NUÔI DỊCH VỤ KỸ THUẬT. CP PHÁT TRIỂN CHĂN NUÔI BẾN VỮNG ĐC: ẤP 2, THUẬN HÒA, LONG MỸ, HẬU GIANG 09666.098.0953 - 0939.053502</corrected>\",
  \"<corrected>ĐẠI LÝ BIA LARUE SÀNH CHUYÊN LOAN SPECIAL CUNG CẤP SỈ VÀ LẺ BIA NƯỚC GIẢI KHÁT ĐT: ẤP 18, THUẬN HƯNG, LONG MỸ, HẬU GIANG 0962671811</corrected>\"
]"""),
            ],
        ),
    ]
    
    user_input = types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=json.dumps(input)),
            ],
        )
    contents.append(user_input)
    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text= 
            """Bạn là một chuyên gia ngôn ngữ được giao nhiệm vụ sửa lỗi văn bản OCR từ bảng hiệu quảng cáo. 
              Văn bản OCR có thể chứa lỗi do thuật toán đôi khi nhận diện sai ký tự, bỏ sót ký tự hoặc thêm 
              ký tự không có trong văn bản gốc. Hãy sử dụng kiến thức ngôn ngữ của bạn để phát hiện và sửa các lỗi này. 
              Lưu ý: 
              1 - Không được bịa đặt thông tin bạn không biết để sửa lỗi.
              2 - Trong phản hồi của bạn, không đưa ra giải thích về nhiệm vụ hoặc kết quả. 
              3 - Chỉ cung cấp văn bản được sửa lỗi. Hãy đưa câu trả lời của bạn vào giữa thẻ <corrected> và </corrected> như trong mã HTML."""),
        ],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    # result = parse_str_2_list(response.text)
    result = (response.text)
    matches = re.findall(r"<corrected>(.*?)</corrected>", result)
    print(matches)
    
if __name__ == "__main__":
    input = []
    generate(input)
