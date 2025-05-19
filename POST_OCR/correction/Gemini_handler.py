import google.generativeai as genai

import enum
from pydantic import BaseModel

# config = {
#     'response_mime_type': 'application/json',
#     'response_schema': Text,
# }
generation_config = {
  "temperature": 0.11,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
class Text(BaseModel):
  corrected_text: str
genai.configure(api_key="AIzaSyAT59AiN0NCWnNkJz5dJiknD9Ghwa7plpE")

# client = genai.Client(api_key="AIzaSyAT59AiN0NCWnNkJz5dJiknD9Ghwa7plpE")
# response = client.models.generate_content(
#     model='gemini-2.0-flash',
#     contents='List 10 home-baked cookie recipes and give them grades based on tastiness.',
#     config={
#         'response_mime_type': 'application/json',
#         'response_schema': Text,
#     },
# )
model = genai.GenerativeModel(
  model_name="gemini-1.5-flash-001",
  generation_config=generation_config,
  system_instruction="Bạn là một trợ lý ảo chăm sóc khách hàng của công ty vận tải hành khách bằng xe khách, tên của bạn là Dawi.\nBạn có nhiệm vụ trả lời thắc mắc, hỗ trợ khách hàng, cung cấp thông tin về chuyến xe, giá vé, chính sách hoàn vé và giải quyết sự cố.\nBạn cũng có thể trò chuyện phím với khách về các vấn đề ngoài lề.",
)

response = model.generate_content(" xin chào, tôi muốn hỏi về chuyến xe từ Hà Nội đi Đà Nẵng vào ngày 20 tháng 10 năm 2023. Giá vé là bao nhiêu? ")

print(response.text)