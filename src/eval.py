import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, model_name, target_names=['Negative', 'Positive']):
    """
    In ra báo cáo phân loại và vẽ Confusion Matrix sử dụng Seaborn.
    """
    print(f"\n{'='*50}")
    print(f"BÁO CÁO KẾT QUẢ: {model_name.upper()}")
    print(f"{'='*50}")
    
    # 1. Text Report (Precision, Recall, F1)
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # 2. Visual Confusion Matrix (The Seaborn way)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names,
                cbar=False) # Tắt thanh màu bên cạnh cho gọn
                
    plt.ylabel('Thực tế (Actual)')
    plt.xlabel('Dự đoán (Predicted)')
    plt.title(f'Confusion Matrix: {model_name.upper()}')
    
    # tight_layout helps prevent labels from getting cut off in notebooks
    plt.tight_layout() 
    plt.show()