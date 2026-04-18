import joblib
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import custom modules
from preprocess import clean_text
from lstm_model import SentimentLSTM
from lstm_data import IMDBDataset
from eval import evaluate_model

def plot_learning_curves():
    """Đọc số liệu từ file history và vẽ biểu đồ Train/Val"""
    history_path = "../saved_models/lstm_history.joblib"
    if not os.path.exists(history_path):
        print(f"[!] Không tìm thấy {history_path}. Hãy chạy lstm_train.py trước.")
        return

    print("\n[1] Đang vẽ đường cong học tập (Learning Curves)...")
    history = joblib.load(history_path)
    
    os.makedirs("../images", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Biểu đồ Loss
    ax[0].plot(history['train_loss'], label='Train Loss', color='blue', marker='o')
    ax[0].plot(history['val_loss'], label='Validation Loss', color='red', marker='s')
    ax[0].set_title('Biểu đồ Loss qua các Epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)

    # Biểu đồ Accuracy
    ax[1].plot(history['train_acc'], label='Train Accuracy', color='blue', marker='o')
    ax[1].plot(history['val_acc'], label='Validation Accuracy', color='red', marker='s')
    ax[1].set_title('Biểu đồ Accuracy qua các Epochs')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig("../images/lstm_learning_curve.png", dpi=300)
    print("Đã lưu: ../images/lstm_learning_curve.png")

def evaluate_lstm_on_test():
    """Nạp mô hình tốt nhất và vẽ Confusion Matrix trên tập Test"""
    model_path = "../saved_models/best_lstm_model.pt"
    vocab_path = "../saved_models/lstm_word2idx.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        print(f"[!] Thiếu file mô hình hoặc vocab. Hãy kiểm tra lại thư mục saved_models.")
        return

    print("\n[2] Đang nạp mô hình LSTM để đánh giá trên tập Test...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Tải và chia dữ liệu (Phải chia y hệt như trong file train)
    df = pd.read_csv("../data/IMDB Dataset.csv")
    df['clean_text'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    _, X_temp, _, y_temp = train_test_split(df['clean_text'], df['label'], test_size=0.3, random_state=42)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 2. Chuẩn bị DataLoader
    word2idx = joblib.load(vocab_path)
    test_dataset = IMDBDataset(X_test, y_test, word2idx, max_len=200)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 3. Nạp kiến trúc và Trọng số (Weights)
    VOCAB_SIZE = len(word2idx)
    model = SentimentLSTM(VOCAB_SIZE, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=2, bidirectional=True, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 4. Suy luận (Inference)
    print("    Đang dự đoán...")
    lstm_preds = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            outputs = model(texts).squeeze(1)
            preds = torch.round(torch.sigmoid(outputs)).cpu().numpy()
            lstm_preds.extend(preds)

    # 5. Tái sử dụng hàm vẽ Confusion Matrix từ Traditional Pipeline
    print("    Đang xuất Confusion Matrix...")
    evaluate_model(y_test, lstm_preds, model_name="lstm")
    # File sẽ tự động được lưu thành cm_lstm.png nhờ logic có sẵn trong eval.py

if __name__ == "__main__":
    print("="*50)
    print("TRỰC QUAN HÓA KIẾN TRÚC DEEP LEARNING (LSTM)")
    print("="*50)
    
    plot_learning_curves()
    evaluate_lstm_on_test()
    
    print("\nHoàn tất quá trình đánh giá!")