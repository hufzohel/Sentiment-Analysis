import torch
from torch.utils.data import Dataset
from collections import Counter
import joblib
import os

class IMDBDataset(Dataset):
    """
    Lớp Dataset tùy chỉnh cho PyTorch để xử lý dữ liệu Text.
    """
    def __init__(self, texts, labels, word2idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Lấy câu văn tại vị trí idx và cắt thành các từ
        text = self.texts.iloc[idx].split()
        
        # Chuyển chữ thành số. Nếu từ không có trong từ điển, dùng token <UNK> (Unknown)
        seq = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in text]
        
        # Căn chỉnh độ dài (Padding / Truncating)
        if len(seq) < self.max_len:
            # Thêm token <PAD> vào cuối cho đủ độ dài
            seq = seq + [self.word2idx['<PAD>']] * (self.max_len - len(seq))
        else:
            # Cắt bớt nếu câu quá dài
            seq = seq[:self.max_len]
            
        # Trả về dưới dạng Tensor của PyTorch
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels.iloc[idx], dtype=torch.float)

def build_vocabulary(train_texts, max_vocab_size=25000, save_path="../saved_models/lstm_word2idx.joblib"):
    """
    Xây dựng bộ từ điển (Vocabulary) TỪ TẬP TRAIN và lưu lại.
    """
    print("[DATA] Đang xây dựng bộ từ điển (Vocabulary)...")
    words = []
    for text in train_texts:
        words.extend(text.split())

    # Đếm tần suất và lấy những từ phổ biến nhất
    word_counts = Counter(words)
    common_words = word_counts.most_common(max_vocab_size)

    # Khởi tạo từ điển với 2 token đặc biệt
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, count) in enumerate(common_words, start=2):
        word2idx[word] = idx

    # Đảm bảo thư mục tồn tại và lưu file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(word2idx, save_path)
    print(f"[DATA] Đã lưu từ điển gồm {len(word2idx)} từ tại: {save_path}")
    
    return word2idx