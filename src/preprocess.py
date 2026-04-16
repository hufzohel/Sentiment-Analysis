import re

def clean_text(text):
    """
    Hàm làm sạch văn bản cơ bản cho dữ liệu tiếng Anh (như IMDB/Amazon).
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Bỏ các thẻ HTML (rất hay gặp trong review phim/sản phẩm)
    text = re.sub(r'<.*?>', ' ', text)
    
    # 2. Bỏ các ký tự đặc biệt, chỉ giữ lại chữ cái
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Đưa về chữ viết thường và xóa khoảng trắng thừa
    text = text.lower().strip()
    
    # Xóa khoảng trắng lặp lại
    text = re.sub(r'\s+', ' ', text)
    
    return text