import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pretrained_embeddings=None):
        super(SentimentLSTM, self).__init__()
        
        # 1. Lớp Word2Vec (Embedding): Biến chỉ số từ thành vector liên tục
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            # Tùy chọn: self.embedding.weight.requires_grad = False (Nếu muốn đóng băng GloVe, không cho học tiếp)
            

        # 2. Lớp LSTM: Xử lý chuỗi thời gian, ghi nhớ ngữ cảnh
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # 3. Lớp Dropout: Tắt ngẫu nhiên các nơ-ron để chống Overfitting
        self.dropout = nn.Dropout(dropout)
        
        # 4. Lớp Fully Connected (Linear): Ép mảng đặc trưng về 1 giá trị đầu ra (0 hoặc 1)
        # Nhân đôi hidden_dim vì chúng ta dùng Bidirectional (2 chiều kết hợp lại)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, text):
        # text shape: [batch_size, sequence_length]
        
        # Bước 1: Tra bảng Word2Vec
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch_size, sequence_length, embedding_dim]
        
        # Bước 2: Chạy qua bộ nhớ LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # Bước 3: Lấy trạng thái ẩn (hidden state) của bước thời gian cuối cùng
        if self.lstm.bidirectional:
            # Nối trạng thái cuối của chiều tiến (hidden[-2]) và chiều lùi (hidden[-1])
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        # Bước 4: Chạy qua lớp tuyến tính để phân loại
        return self.fc(hidden)