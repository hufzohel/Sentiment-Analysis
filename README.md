# Phân tích Sắc thái (Sentiment Analysis) trên Dữ liệu Văn bản
**Môn học:** Học máy (Machine Learning)  
**Mã môn học:** CO3117  
**Học kỳ:** 252 (Học kỳ II - Năm học 2025-2026)  
**Trường:** Đại học Bách Khoa - ĐHQG-HCM (HCMUT)

---

## 👨‍🏫 Giảng viên hướng dẫn
* **TS. Trương Vĩnh Lân**

## 👥 Danh sách thành viên nhóm
| Họ và tên              | MSSV    | Tỉ lệ đóng góp |
|:-----------------------|:--------| :--- |
| Trần Nguyễn Mạnh Cường | 2210446 | 35% |
| Nguyễn Quang Bảo       | 2410276 | 35% |
| Trần Trung Kiên        | 2311744 | 10% |
| Nguyễn Đăng Khoa       | 2211617 | 10% |
| Nguyễn Trọng Nghĩa     | 2312271 | 10% |
---

## 🎯 Mục tiêu bài tập lớn
Dự án này thực hiện một pipeline học máy hoàn chỉnh để giải quyết bài toán **Phân tích Sắc thái (Sentiment Analysis)**. Các mục tiêu chính bao gồm:
1. **Pipeline truyền thống:** Thực hiện EDA, Tiền xử lý, Trích xuất đặc trưng (TF-IDF) và huấn luyện các mô hình như SVM, Naive Bayes, Logistic Regression.
2. **Mở rộng Học sâu (Deep Learning):** So sánh phương pháp truyền thống với mô hình LSTM + Word2vec và các mô hình Transformer hiện đại (BERT/RoBERTa) để đánh giá hiệu suất trên dữ liệu văn bản phức tạp.

## 📂 Cấu trúc thư mục
Tuân thủ theo yêu cầu của đề bài:
* `saved_models`: Các model đã được train và lưu để có thể được kiểm chứng nhanh chóng (thông thường sẽ bị gitignore nhưng ở đây chúng em lưu lại lên github một phần để dễ kiểm chứng một phần đóng vai trò làm automatic version control tương tự như YOLOv8)
* `src/`: Các script Python (.py) cho pipeline truyền thống.
* `deep_learning_src/`: Các script Python (.py) cho pipeline deep learning.  
* `data/`: Dataset và Glove text file (không cần thiết đối với notebook vì script trên notebook tự cài đặt dataset và glove).
* `notebook/`: Chứa file .ipynb để huấn luyện và đánh giá mô hình.
* `report/`: Chứa file báo cáo.
* `images/`: Hình ảnh EDA (phân phối nhãn dữ liệu, phân phối độ dài văn bản, tần suất xuất hiện từ) và biểu đồ đánh giá mô hình.
* `feature/`: Chứa các tệp tin đặc trưng đã được trích xuất hoặc các ma trận nhúng (embedding matrices) phục vụ cho quá trình huấn luyện.
* `requirements.txt`: Danh sách các thư viện cần cài đặt.
* `README.md`

## 🚀 Hướng dẫn chạy chương trình
1. Mở file notebook tại link: (https://colab.research.google.com/drive/1iLm98k_Gi0H5u7AO1TJJ_i_SXxkZXfUe?usp=sharing).
2. Chọn loại kết nối là T4 GPU
3. Chọn **Runtime > Run all** (hoặc Runtime > Chạy tất cả).
4. Notebook sẽ tự động:
   - Cài đặt các thư viện cần thiết (`transformers`, `scikit-learn`,...).
   - Tải tập dữ liệu từ nguồn công khai trực tiếp vào môi trường Colab.
   - Thực hiện quy trình EDA, huấn luyện và đánh giá mô hình.

## 📊 Tóm tắt kết quả
<img width="1389" height="489" alt="output_comparison_all" src="https://github.com/user-attachments/assets/fb85bf66-dc42-4901-86c1-7f821ea588c9" />


## 💻 Hướng dẫn chạy chương trình tại Local

Nếu muốn thực thi dự án trên máy tính cá nhân thay vì Google Colab, hãy thực hiện theo các bước sau:

### 1. Yêu cầu hệ thống
*   **Python:** Phiên bản 3.9 trở lên.
*   **GPU (Khuyến nghị):** NVIDIA GPU với hỗ trợ CUDA để huấn luyện các mô hình Deep Learning (LSTM, BERT). Nếu chạy trên CPU, thời gian huấn luyện sẽ rất lâu.
*   **Bộ nhớ:** Trống ít nhất **5GB** để lưu trữ dataset và các mô hình pre-trained.

### 2. Cài đặt môi trường
Mở terminal (hoặc Command Prompt) và thực hiện:
```bash
# 1. Clone project từ GitHub
git clone <url_cua_nhom>
cd <ten_thu_muc_du_an>

# 2. Khởi tạo môi trường ảo (venv)
python -m venv venv

# 3. Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên Linux/macOS:
source venv/bin/activate

# 4. Cài đặt các thư viện cần thiết
pip install --upgrade pip
pip install -r requirements.txt
```
### 3. Chuẩn bị dữ liệu (GloVe Embeddings)
Vui lòng tải các file sau và đặt vào thư mục `data/` ở thư mục gốc của dự án:

Pre-trained GloVe Embeddings (Stanford NLP):
   - Truy cập trang chủ: [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
   - Tải gói `glove.6B.zip` (822 MB).
   - Giải nén và copy file wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined bỏ vào thư mục `data/`.