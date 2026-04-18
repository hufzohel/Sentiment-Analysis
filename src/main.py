import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Import custom modules
from preprocess import clean_text
from models import build_pipeline
from eval import evaluate_model

# ---------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ---------------------------------------------------------
print("Đang đọc dữ liệu local...")
df = pd.read_csv("../data/IMDB Dataset.csv") 

print("Đang tiền xử lý văn bản...")
df['clean_text'] = df['review'].apply(clean_text)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 2. THE COMPARISON LOOP (TRAIN, SAVE & EVALUATE)
# ---------------------------------------------------------
models_to_compare = ["naive_bayes", "logistic", "svm"]

# Tạo thư mục lưu model nếu chưa có
os.makedirs("../saved_models", exist_ok=True)
os.makedirs("../images", exist_ok=True)

# Từ điển để lưu điểm số phục vụ vẽ biểu đồ so sánh
f1_scores = {}

for model_name in models_to_compare:
    tuned_model_path = "../saved_models/best_linearsvc_tuned.joblib"
    
    # 1. Xác định tên hiển thị trước
    is_svm_tuned = (model_name == "svm" and os.path.exists(tuned_model_path))
    display_name = "SVM (Tuned)" if is_svm_tuned else model_name.upper()

    # 2. Load hoặc Train
    if is_svm_tuned:
        print(f"\n[VIP] Đang nạp SVM ĐÃ TINH CHỈNH cho biểu đồ so sánh...")
        pipeline = joblib.load(tuned_model_path)
    else:
        print(f"\n[INFO] Đang huấn luyện mô hình baseline: {model_name.upper()}...")
        pipeline = build_pipeline(model_name=model_name, max_features=5000)
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, f"../saved_models/{model_name}_baseline.joblib")

    # 3. Dự đoán và tính điểm
    y_pred = pipeline.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    
    # Lưu vào dictionary với KEY là display_name (để vẽ biểu đồ dùng key này)
    f1_scores[display_name] = score
    
    # Gọi hàm đánh giá với display_name để lưu CM đúng tên
    evaluate_model(y_test, y_pred, model_name=display_name)

# ---------------------------------------------------------
# 3. TRỰC QUAN HÓA SO SÁNH 3 MÔ HÌNH (CHO BÁO CÁO)
# ---------------------------------------------------------
print("\n[INFO] Đang vẽ biểu đồ so sánh 3 mô hình...")
plt.figure(figsize=(8, 5))
sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()), palette='viridis')

plt.title('So sánh hiệu năng (F1-Score) của 3 Mô hình cơ sở')
plt.ylabel('F1-Score (Macro)')
plt.xlabel('Mô hình')
plt.ylim(0.7, 1.0) # Thu hẹp trục Y để thấy rõ sự khác biệt

# Điền số liệu trực tiếp lên cột
for index, value in enumerate(f1_scores.values()):
    plt.text(index, value + 0.01, f'{value:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("../images/model_comparison.png", dpi=300)
print("[OK] Đã lưu biểu đồ so sánh tại: ../images/model_comparison.png")