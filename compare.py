import pandas as pd
import numpy as np
import torch
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from preprocess import clean_text
from torch.utils.data import DataLoader


from deep_learning_src.lstm_model import SentimentLSTM
from deep_learning_src.lstm_data import IMDBDataset

def run_comparison():
    print("="*50)
    print("ĐẤU TRƯỜNG SO SÁNH: TRADITIONAL ML vs DEEP LEARNING 🥊")
    print("="*50)

    # 1. TẢI TẬP TEST CHUNG (Phải chia y hệt như lúc train để đảm bảo công bằng)
    print("\n[1] Đang tải tập dữ liệu Test...")
    df = pd.read_csv("data/IMDB Dataset.csv")
    df['clean_text'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X_train, X_temp, y_train, y_temp = train_test_split(df['clean_text'], df['label'], test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    results = {} # Lưu kết quả để vẽ hình

    # -------------------------------------------------------------------
    # 2. ĐÁNH GIÁ MÔ HÌNH 1: LINEARSVC (TUNED)
    # -------------------------------------------------------------------
    svm_path = "../saved_models/best_linearsvc_tuned.joblib"
    if os.path.exists(svm_path):
        print(f"\n[2] Đang nạp và đánh giá mô hình ML (SVM)...")
        svm_model = joblib.load(svm_path)
        svm_preds = svm_model.predict(X_test)
        
        results['SVM (Tuned)'] = {
            'Accuracy': accuracy_score(y_test, svm_preds),
            'F1 Score (Macro)': f1_score(y_test, svm_preds, average='macro')
        }
    else:
        print("\n[!] Không tìm thấy mô hình SVM Tuned.")

    # -------------------------------------------------------------------
    # 3. ĐÁNH GIÁ MÔ HÌNH 2: Bi-LSTM (GLOVE)
    # -------------------------------------------------------------------
    lstm_path = "saved_models/best_lstm_model.pt"
    vocab_path = "saved_models/lstm_word2idx.joblib"
    
    if os.path.exists(lstm_path) and os.path.exists(vocab_path):
        print(f"\n[3] Đang nạp và đánh giá mô hình Deep Learning (LSTM)...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load từ điển và chuẩn bị DataLoader cho tập Test
        word2idx = joblib.load(vocab_path)
        test_dataset = IMDBDataset(X_test, y_test, word2idx, max_len=200)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Load mô hình LSTM (Lưu ý: Không cần truyền GloVe vào nữa vì trọng số đã được lưu trong file .pt)
        VOCAB_SIZE = len(word2idx)
        lstm_model = SentimentLSTM(VOCAB_SIZE, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=2, bidirectional=True, dropout=0.5)
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
        lstm_model = lstm_model.to(device)
        lstm_model.eval()
        
        lstm_preds = []
        with torch.no_grad():
            for texts, labels in test_loader:
                texts = texts.to(device)
                outputs = lstm_model(texts).squeeze(1)
                preds = torch.round(torch.sigmoid(outputs)).cpu().numpy()
                lstm_preds.extend(preds)
                
        results['Bi-LSTM (GloVe)'] = {
            'Accuracy': accuracy_score(y_test, lstm_preds),
            'F1 Score (Macro)': f1_score(y_test, lstm_preds, average='macro')
        }
    else:
        print("\n[!] Không tìm thấy mô hình LSTM hoặc file Vocab.")

    # -------------------------------------------------------------------
    # 4. TRỰC QUAN HÓA KẾT QUẢ SO SÁNH
    # -------------------------------------------------------------------
    if len(results) == 2:
        print("\n[4] Đang vẽ biểu đồ so sánh đại chiến...")
        # Chuyển Dictionary thành DataFrame để vẽ cho dễ
        plot_df = pd.DataFrame(results).T.reset_index()
        plot_df = plot_df.melt(id_vars='index', var_name='Metric', value_name='Score')
        plot_df.rename(columns={'index': 'Model'}, inplace=True)

        plt.figure(figsize=(9, 6))
        ax = sns.barplot(x='Model', y='Score', hue='Metric', data=plot_df, palette='Set2')
        
        plt.title('So sánh Hiệu năng: Học máy Truyền thống vs Học Sâu', fontsize=14, fontweight='bold')
        plt.ylim(0.8, 1.0) # Thu hẹp trục Y để thấy rõ sự khác biệt (Điều chỉnh nếu cần)
        plt.ylabel('Điểm số')
        
        # Ghi số trực tiếp lên đầu cột
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 5), textcoords='offset points')

        plt.tight_layout()
        os.makedirs("../images", exist_ok=True)
        plt.savefig("../images/ultimate_comparison.png", dpi=300)
        print("Đã lưu biểu đồ đại chiến tại: ../images/ultimate_comparison.png")
    else:
        print("Chưa đủ 2 mô hình để vẽ biểu đồ. Hãy đảm bảo bạn đã train xong cả SVM và LSTM.")

if __name__ == "__main__":
    run_comparison()