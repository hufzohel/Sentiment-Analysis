# Phân tích Sắc thái (Sentiment Analysis) trên Dữ liệu Văn bản
**Môn học:** Học máy (Machine Learning)  
**Mã môn học:** CO3117  
**Học kỳ:** 252 (Học kỳ II - Năm học 2025-2026)  
**Trường:** Đại học Bách Khoa - ĐHQG-HCM (HCMUT)

---

## 👨‍🏫 Giảng viên hướng dẫn
* **TS. Trương Vĩnh Lân**

## 👥 Danh sách thành viên nhóm
| Họ và tên | MSSV | Tỉ lệ đóng góp |
| :--- | :--- | :--- |
| Trần Nguyễn Mạnh Cường | 2210446 | 35% |
| Nguyễn Trọng Nghĩa | 2312271 | 35% |
| Trần Trung Kiên | 2311744 | 10% |
| Nguyễn Đăng Khoa | 2211617 | 10% |
| Nguyễn Quang Bảo | 2410276 | 10% |
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

