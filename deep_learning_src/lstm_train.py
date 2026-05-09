import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import joblib
# from torchtext.vocab import GloVe
import numpy as np

# Import các module custom của bạn
from preprocess import clean_text
from lstm_model import SentimentLSTM
from lstm_data import IMDBDataset, build_vocabulary

# ---------------------------------------------------------
# 1. SIÊU THAM SỐ (HYPERPARAMETERS)
# ---------------------------------------------------------
MAX_VOCAB_SIZE = 25000
MAX_SEQ_LEN = 200      
BATCH_SIZE = 64
EMBEDDING_DIM = 100    
HIDDEN_DIM = 256       
N_LAYERS = 2           
BIDIRECTIONAL = True
DROPOUT = 0.5
EPOCHS = 10            
LEARNING_RATE = 0.001
L2REG = 1e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" Đang chạy phần cứng: {device}")

# ---------------------------------------------------------
# 2. CHUẨN BỊ VÀ CHIA DỮ LIỆU (TRAIN / VAL / TEST)
# ---------------------------------------------------------
print("\n[1/4] Đang đọc và làm sạch dữ liệu...")
df = pd.read_csv("../data/IMDB Dataset.csv")
df['clean_text'] = df['review'].apply(clean_text)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_temp, y_train, y_temp = train_test_split(
    df['clean_text'], df['label'], test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"  -> Tập Train: {len(X_train)} câu")
print(f"  -> Tập Val:   {len(X_val)} câu")
print(f"  -> Tập Test:  {len(X_test)} câu (Cất vào két sắt)")

# ---------------------------------------------------------
# 3. TỪ ĐIỂN VÀ DATALOADER
# ---------------------------------------------------------
print("\n[2/4] Thiết lập Dữ liệu vào PyTorch...")
word2idx = build_vocabulary(X_train, MAX_VOCAB_SIZE, "../saved_models/lstm_word2idx.joblib")

print("\n[2.5/4] Đang nạp Pre-trained GloVe bằng Python thuần...")
glove_dict = {}
glove_path = "../data/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"

# Check xem đã tải file về chưa, chưa tải thì chửi luôn cho nhanh
if not os.path.exists(glove_path):
    print(f"\n LỖI TRÍ MẠNG: Không tìm thấy file {glove_path}!")
    print("Vui lòng lên mạng tải file glove.6B.100d.txt (từ bộ GloVe Stanford) và ném vào thư mục data/")
    exit()

# Mở file txt lên và nhặt từng từ bỏ vào Dictionary
with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        # values[1:] chứa 100 con số, ép kiểu về float32 của numpy
        try:
            # Cố gắng ép kiểu về số thực
            vector = np.asarray(values[1:], dtype='float32')
            
            # Đảm bảo vector có đúng 100 chiều (100d) thì mới lấy
            if len(vector) == 100:
                glove_dict[word] = vector
        except ValueError:
            # Nếu gặp dòng lỗi (chứa ký tự lạ, dấu chấm...), bỏ qua luôn
            continue
        vector = np.asarray(values[1:], dtype='float32')
        glove_dict[word] = vector

print("[3/4] Đang đóng gói dữ liệu và khớp từ điển với GloVe...")
VOCAB_SIZE = len(word2idx)
pretrained_embeddings = torch.zeros(VOCAB_SIZE, 100)

found_words = 0
for word, idx in word2idx.items():
    if word in glove_dict: 
        # Nếu từ của mình CÓ trong từ điển của Stanford thì lấy vector của nó
        pretrained_embeddings[idx] = torch.tensor(glove_dict[word])
        found_words += 1
    else:
        # Từ lóng, sai chính tả... thì đành random vậy
        # Thu nhỏ tiếng ồn lại để không át mất GloVe
        pretrained_embeddings[idx] = torch.normal(mean=0, std=0.1, size=(100,))

print(f"   -> Đã khớp thành công {found_words}/{VOCAB_SIZE} từ vựng từ GloVe!")

# ---------------------------------------------------------
# [Phần tiếp theo: Khởi tạo model và đưa pretrained_embeddings vào]
print(f"\n[4/4] Khởi tạo kiến trúc Bi-LSTM với GloVe Embeddings...")
model = SentimentLSTM(
    vocab_size=VOCAB_SIZE, 
    embedding_dim=100, 
    hidden_dim=256, 
    output_dim=1, 
    n_layers=2, 
    bidirectional=True, 
    dropout=0.5,
    pretrained_embeddings=pretrained_embeddings  
)
model = model.to(device)

train_dataset = IMDBDataset(X_train, y_train, word2idx, MAX_SEQ_LEN)
val_dataset = IMDBDataset(X_val, y_val, word2idx, MAX_SEQ_LEN)
test_dataset = IMDBDataset(X_test, y_test, word2idx, MAX_SEQ_LEN) 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------
# 4. THIẾT LẬP HÀM MẤT MÁT VÀ TỐI ƯU HÓA
# ---------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2REG)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

# ---------------------------------------------------------
# 5. VÒNG LẶP HUẤN LUYỆN (TRAINING ENGINE)
# ---------------------------------------------------------
print("\n[5/5] BẮT ĐẦU HUẤN LUYỆN...")
best_valid_loss = float('inf')
os.makedirs("../saved_models", exist_ok=True)

PATIENCE = 3
early_stop_counter = 0

history_train_loss, history_val_loss = [], []
history_train_acc, history_val_acc = [], []

for epoch in range(EPOCHS):
    # --- PHASE 1: TRAIN ---
    model.train()
    epoch_train_loss, epoch_train_acc = 0, 0
    
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(texts).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
        epoch_train_acc += acc.item()

    train_loss = epoch_train_loss / len(train_loader)
    train_acc = epoch_train_acc / len(train_loader)

    # --- PHASE 2: VALIDATION ---
    model.eval()
    epoch_val_loss, epoch_val_acc = 0, 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            
            epoch_val_loss += loss.item()
            epoch_val_acc += acc.item()
            
    val_loss = epoch_val_loss / len(val_loader)
    val_acc = epoch_val_acc / len(val_loader)

    print(f"Epoch {epoch+1:02}/{EPOCHS} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%")

    history_train_loss.append(train_loss)
    history_val_loss.append(val_loss)
    history_train_acc.append(train_acc)
    history_val_acc.append(val_acc)

    # --- EARLY STOPPING / SAVE BEST MODEL ---
    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), "../saved_models/best_lstm_model.pt")
        print(f"   => [LƯU MODEL] Validation Loss giảm, đã lưu checkpoint!")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"   => [CẢNH BÁO] Val Loss không giảm. Patience: {early_stop_counter}/{PATIENCE}")
        if early_stop_counter >= PATIENCE:
            print("\n🛑 KÍCH HOẠT EARLY STOPPING! Ngăn chặn Overfitting thành công.")
            break

print("\n🎉 HUẤN LUYỆN HOÀN TẤT!")

# ---------------------------------------------------------
# 6. XUẤT STATE TRACKING CHO FILE EVAL
# ---------------------------------------------------------
history = {
    'train_loss': history_train_loss,
    'val_loss': history_val_loss,
    'train_acc': history_train_acc,
    'val_acc': history_val_acc
}

joblib.dump(history, "../saved_models/lstm_history.joblib")
print("Đã lưu lịch sử số liệu huấn luyện tại: ../saved_models/lstm_history.joblib")
print("Gõ 'python lstm_eval.py' để vẽ biểu đồ Learning Curve và Confusion Matrix nhé!")