import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        # Sử dụng hàm __call__ của tokenizer để tránh lỗi AttributeError
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }


def get_loaders(file_path, tokenizer, batch_size=32, max_len=128):
    df = pd.read_csv(file_path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review'].values, df['sentiment'].values, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(IMDBDataset(train_texts, train_labels, tokenizer, max_len), batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(IMDBDataset(val_texts, val_labels, tokenizer, max_len), batch_size=batch_size)

    return train_loader, val_loader