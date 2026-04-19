import os
import sys
import torch
import warnings
from torch.optim import AdamW
from transformers import BertTokenizer
from tqdm import tqdm

# Cấu hình môi trường và ẩn cảnh báo
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import logging
from transformers import logging as transformers_logging

# Chỉ hiện lỗi (Error), không hiện cảnh báo (Warning) màu đỏ nữa
transformers_logging.set_verbosity_error()

# Fix đường dẫn module nếu chạy từ thư mục con
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bert_data import get_loaders
from bert_model import get_bert_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = "../saved_models/bert_imdb.pth"


def train():
    os.makedirs("../saved_models", exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Gọi đúng hàm get_loaders từ bert_data
    train_loader, _ = get_loaders("../data/IMDB Dataset.csv", tokenizer)

    model = get_bert_model().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    epochs = 3
    print(f"Bắt đầu huấn luyện trên thiết bị: {DEVICE}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"\n>>> Kết thúc Epoch {epoch} - Average Loss: {avg_loss:.4f}\n")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Huấn luyện hoàn tất! Model đã được lưu tại: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()