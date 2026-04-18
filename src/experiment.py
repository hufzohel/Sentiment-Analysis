import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from preprocess import clean_text
from eval import evaluate_model
# ---------------------------------------------------------
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ---------------------------------------------------------
print("Đang đọc dữ liệu local...")
df = pd.read_csv("../data/IMDB Dataset.csv")

# ĐÃ FIX: Sửa 'text' thành 'review'
df['clean_text'] = df['review'].apply(clean_text)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 2. XÂY DỰNG PIPELINE VÀ CHẠY GRIDSEARCHCV
# ---------------------------------------------------------
base_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('classifier', LinearSVC(random_state=42, dual=False, max_iter=2000))
])

param_grid = {
    'tfidf__max_features': [5000, 7000],  
    'classifier__C': [0.1, 1.0, 10.0]     
}

print("\n[INFO] Đang khởi chạy Grid Search cho LinearSVC... (Khoảng 1-3 phút)")
grid_search = GridSearchCV(
    estimator=base_pipeline,
    param_grid=param_grid,
    cv=5,               
    scoring='f1',       
    n_jobs=-1,          
    verbose=2           
)

grid_search.fit(X_train, y_train)

# ---------------------------------------------------------
# 3. BÁO CÁO VÀ LƯU MODEL TỐI ƯU
# ---------------------------------------------------------
print("\n" + "="*50)
print("TỐI ƯU HÓA LINEARSVC HOÀN TẤT!")
print("Bộ tham số xịn nhất tìm được:", grid_search.best_params_)
print(f"Điểm F1 trung bình trên tập Validation: {grid_search.best_score_:.4f}")
print("="*50)

best_model = grid_search.best_estimator_

# LƯU LẠI MODEL XỊN NHẤT NÀY VÀO THƯ MỤC
os.makedirs("../saved_models", exist_ok=True)
joblib.dump(best_model, "../saved_models/best_linearsvc_tuned.joblib")
print(f"[OK] Đã lưu mô hình tối ưu tại: ../saved_models/best_linearsvc_tuned.joblib")

from sklearn.metrics import classification_report
y_pred = best_model.predict(X_test)
print("\nBáo cáo trên tập Test (Dữ liệu chưa từng nhìn thấy):")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

evaluate_model(y_test, y_pred, model_name="svm_tuned")