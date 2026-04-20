import re

def clean_text(text):
    """
    Hàm làm sạch văn bản tối ưu cho Sentiment Analysis.
    Giữ lại trọn vẹn ý nghĩa của các câu phủ định.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Bỏ các thẻ HTML (rất hay gặp trong review phim/sản phẩm)
    text = re.sub(r'<.*?>', ' ', text)
    
    # 2. BƯỚC MỚI: Tách các từ viết tắt (Contraction Expansion)
    # Tách các cụm phủ định để mô hình "nhìn" thấy chữ "not"
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    
    # 3. Bỏ các ký tự đặc biệt, NHƯNG thêm dấu nháy đơn (') vào danh sách an toàn
    # Regex mới: [^a-zA-Z\s\'] -> Cho phép A-Z, a-z, khoảng trắng và dấu '
    text = re.sub(r"[^a-zA-Z\s\']", '', text)
    
    # 4. Đưa về chữ viết thường và xóa khoảng trắng thừa
    text = text.lower().strip()
    
    # 5. Xóa khoảng trắng lặp lại
    text = re.sub(r'\s+', ' ', text)
    
    return text