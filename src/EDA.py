import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

def plot_class_distribution(df, label_col='sentiment'):
    """
    Trực quan hóa độ cân bằng của tập dữ liệu (Class Distribution).
    """
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=label_col, data=df, palette='Set2')
    plt.title('Phân phối nhãn dữ liệu (Class Distribution)')
    plt.xlabel('Sắc thái (Sentiment)')
    plt.ylabel('Số lượng mẫu')
    
    # Thêm số liệu trực tiếp lên cột
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig('../images/class_distribution.png', dpi=300)
    plt.show()

def plot_text_length(df, text_col='clean_text', label_col='sentiment'):
    """
    Trực quan hóa phân phối chiều dài văn bản giữa các lớp.
    Giúp quyết định độ dài padding cho mô hình Deep Learning (LSTM/BERT).
    """
    # Tính số từ trong mỗi câu
    df['word_count'] = df[text_col].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='word_count', hue=label_col, bins=50, kde=True, palette='Set2')
    plt.title('Phân phối chiều dài văn bản (Word Count Distribution)')
    plt.xlabel('Số lượng từ (Tokens)')
    plt.ylabel('Tần suất')
    plt.xlim(0, 1000) # Cắt biểu đồ ở 1000 từ để dễ nhìn
    plt.tight_layout()
    plt.savefig('../images/text_length.png', dpi=300)
    plt.show()

def plot_word_clouds(df, text_col='clean_text', label_col='sentiment'):
    """
    Vẽ Word Cloud để xem các từ xuất hiện nhiều nhất trong Tích cực và Tiêu cực.
    """
    positive_text = " ".join(text for text in df[df[label_col] == 'positive'][text_col])
    negative_text = " ".join(text for text in df[df[label_col] == 'negative'][text_col])
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    # Wordcloud Tích cực
    wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_text)
    ax[0].imshow(wordcloud_pos, interpolation='bilinear')
    ax[0].set_title('Từ vựng phổ biến: Tích cực (Positive)', fontsize=16)
    ax[0].axis('off')
    
    # Wordcloud Tiêu cực
    wordcloud_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_text)
    ax[1].imshow(wordcloud_neg, interpolation='bilinear')
    ax[1].set_title('Từ vựng phổ biến: Tiêu cực (Negative)', fontsize=16)
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('../images/word_clouds.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    import pandas as pd
    import os
    from preprocess import clean_text
    
    print("Đang nạp dữ liệu local cho EDA...")
    df = pd.read_csv("../data/IMDB Dataset.csv")
    df['clean_text'] = df['review'].apply(clean_text)
    
    # Tạo thư mục lưu ảnh báo cáo nếu chưa có
    if not os.path.exists("images"):
        os.makedirs("images")
        
    print("Vẽ phân phối nhãn...")
    plot_class_distribution(df)
    
    print("Vẽ phân phối chiều dài...")
    plot_text_length(df)
    
    print("Vẽ Word Cloud...")
    plot_word_clouds(df)
    
    print("Đã lưu toàn bộ ảnh vào thư mục 'images/'!")