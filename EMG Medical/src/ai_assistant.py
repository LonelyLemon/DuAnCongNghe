import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY không được tìm thấy trong biến môi trường")

genai.configure(api_key=API_KEY)

APP_MANUAL = """
THÔNG TIN VỀ ỨNG DỤNG "EMG LAB PRO":
1. Chức năng chính: Quản lý và phân tích tín hiệu điện cơ (EMG) từ máy Natus.
2. Tải dữ liệu: Kéo thả file .txt vào khung upload ở Trang chủ. Hỗ trợ import nhiều file cùng lúc.
3. Tìm kiếm & Quản lý: 
   - Ô tìm kiếm ở trang chủ dùng để lọc bệnh nhân theo tên.
   - Có thể sửa thông tin hành chính hoặc xóa bản ghi bằng các nút tương ứng.
4. Màn hình Phân tích (Analysis):
   - Biểu đồ: Dùng chuột kéo để zoom, click đúp để reset.
   - Bộ lọc (DSP): Nằm ở góc trái, gồm "Notch 50Hz" (khử nhiễu nguồn) và "Bandpass 20-500Hz".
   - Xem thông số kỹ thuật: Bấm nút "ℹ️ Thông số kỹ thuật" ở panel thông tin bệnh nhân.
5. Gán nhãn (Labeling):
   - Bước 1: Chọn loại nhãn ở Dropdown (VD: MYO, NEURO...).
   - Bước 2: Dùng chuột vẽ một vùng hình chữ nhật (Box Select) bao quanh đoạn tín hiệu trên biểu đồ.
   - Bước 3: Bấm "Lưu vùng chọn". Hệ thống tự tính P2P, RMS.
   - Có thể tạo loại nhãn mới bằng nút dấu cộng (+).
6. Xuất báo cáo: Bấm nút "Xuất Báo Cáo PDF" màu cam để tải file kết quả.
7. Kết luận: Nhập văn bản vào ô "Kết luận lâm sàng" ở cuối trang và bấm Lưu.
"""

def ask_gemini_medical(user_question, context_data=None):
    try:
        # 1. Prompt
        system_instruction = f"""
        Bạn là Trợ lý thông minh của phần mềm "EMG Lab Pro".
        Vai trò của bạn có 2 nhiệm vụ chính:
        
        NHIỆM VỤ 1: Hỗ trợ chuyên môn Y khoa (EMG/NCS)
        - Giải thích thuật ngữ, nhận xét tín hiệu dựa trên dữ liệu thống kê được cung cấp.
        - Trả lời ngắn gọn, chuyên nghiệp.
        
        NHIỆM VỤ 2: Hướng dẫn sử dụng phần mềm
        - Dựa vào [TÀI LIỆU HƯỚNG DẪN] dưới đây để chỉ dẫn người dùng cách thao tác.
        - Nếu người dùng hỏi tính năng không có trong tài liệu, hãy nói là ứng dụng chưa hỗ trợ.
        
        [TÀI LIỆU HƯỚNG DẪN]:
        {APP_MANUAL}
        
        LƯU Ý: Trả lời bằng tiếng Việt. Giọng điệu thân thiện, hữu ích.
        """
        
        context_str = ""
        if context_data:
            context_str = f"\n\n[DỮ LIỆU TÍN HIỆU HIỆN TẠI]:\n{context_data}\n"

        full_prompt = f"{system_instruction}{context_str}\n\nNgười dùng hỏi: {user_question}"

        # 2. Model
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Vui lòng kiểm tra lại API Key hoặc AI Model: {str(e)}"
        
    except Exception as e:
        return f"Lỗi kết nối AI: {str(e)}"
    
if __name__ == "__main__":
    for model in genai.list_models():
        print(model.name)