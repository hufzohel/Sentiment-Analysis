import torch
import sys, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import BertTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bert_data import get_loaders
from bert_model import get_bert_model


def save_metrics_images(y_true, y_pred, output_dir="../images"):
    """Hàm vẽ và lưu Confusion Matrix và Classification Report thành ảnh."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Vẽ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix - BERT Sentiment Analysis')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    # 2. Lưu Classification Report dưới dạng bảng ảnh
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 4))
    sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap='YlGnBu', cbar=False)
    plt.title('Classification Report Metrics')
    plt.savefig(f"{output_dir}/classification_report.png")
    plt.close()

    print(f"--- Đã lưu ảnh vào thư mục: {os.path.abspath(output_dir)} ---")


def evaluate():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    _, val_loader = get_loaders("../data/IMDB Dataset.csv", tokenizer)

    model = get_bert_model().to(DEVICE)

    model_path = "../saved_models/bert_imdb.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Không tìm thấy file model tại {model_path}!")
        return

    model.eval()
    y_pred, y_true = [], []

    print("Đang đánh giá trên tập validation...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels']

            outputs = model(input_ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=1)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    # In kết quả ra console
    print("\nKẾT QUẢ ĐÁNH GIÁ:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

    # Lưu kết quả thành ảnh
    save_metrics_images(y_true, y_pred)


if __name__ == "__main__":
    evaluate()