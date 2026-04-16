import pandas as pd
from sklearn.model_selection import train_test_split

# Import your custom modules
from data_loader import load_dataset
from preprocess import clean_text
from models import build_pipeline
from eval import evaluate_model

# ---------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ---------------------------------------------------------
print("Đang tải dữ liệu...")
url_imdb = "https://raw.githubusercontent.com/laxmimerit/IMDB-Movie-Reviews-Dataset/master/train.csv"
df = load_dataset(url_imdb, filename="imdb_train.csv")

print("Đang tiền xử lý văn bản...")
df['clean_text'] = df['text'].apply(clean_text)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 2. THE COMPARISON LOOP (TRAIN & EVALUATE)
# ---------------------------------------------------------
# Define the models we want to compare from our factory
models_to_compare = ["naive_bayes", "logistic", "svm"]

for model_name in models_to_compare:
    print(f"\n[INFO] Đang khởi tạo và huấn luyện mô hình: {model_name.upper()}...")
    
    # Instantiate the architecture from models.py
    pipeline = build_pipeline(model_name=model_name, max_features=5000)
    
    # Train (Fits TF-IDF vocabulary and trains the classifier weights)
    pipeline.fit(X_train, y_train)
    
    # Inference
    print(f"[INFO] Đang dự đoán với {model_name.upper()}...")
    y_pred = pipeline.predict(X_test)
    
    # Evaluate (Prints report and plots Confusion Matrix)
    evaluate_model(y_test, y_pred, model_name=model_name)