# Phân tích Sắc thái (Sentiment Analysis) trên Dữ liệu Văn bản
**Môn học:** Học máy (Machine Learning)  
**Mã môn học:** CO3117  
**Học kỳ:** 252 (Học kỳ II - Năm học 2025-2026)  
**Trường:** Đại học Bách Khoa - ĐHQG-HCM (HCMUT)

---

## 👨‍🏫 Giảng viên hướng dẫn
* **TS. Trương Vĩnh Lân**

## 👥 Danh sách thành viên nhóm
| Họ và tên | MSSV | Email | Tỉ lệ đóng góp |
| :--- | :--- | :--- | :--- |
| [Tên của bạn] | [MSSV] | [Email] | 33.3% |
| [Thành viên 2] | [MSSV] | [Email] | 33.3% |
| [Thành viên 3] | [MSSV] | [Email] | 33.3% |

---

## 🎯 Mục tiêu bài tập lớn
Dự án này thực hiện một pipeline học máy hoàn chỉnh để giải quyết bài toán **Phân tích Sắc thái (Sentiment Analysis)**. Các mục tiêu chính bao gồm:
1. **Pipeline truyền thống:** Thực hiện EDA, Tiền xử lý, Trích xuất đặc trưng (TF-IDF/Bag-of-Words) và huấn luyện các mô hình như SVM, Naive Bayes, Logistic Regression.
2. **Mở rộng Học sâu (Deep Learning):** So sánh phương pháp truyền thống với các mô hình Transformer hiện đại (BERT/RoBERTa) để đánh giá hiệu suất trên dữ liệu văn bản phức tạp.

## 📂 Cấu trúc thư mục
Tuân thủ theo yêu cầu của đề bài:
* `notebooks/`: Chứa các file Google Colab chính (.ipynb) để thực hiện EDA và huấn luyện mô hình.
* `modules/`: Các script Python (.py) hỗ trợ tiền xử lý và trích xuất đặc trưng.
* `reports/`: Báo cáo cuối kỳ định dạng PDF và các hình ảnh trực quan hóa dữ liệu.
* `features/`: Các file đặc trưng/embeddings đã trích xuất, lưu dưới dạng `.npy` hoặc `.h5`.
* `data/`: Script tự động tải dữ liệu từ nguồn công khai (không lưu trữ dữ liệu trực tiếp trên repo).

## 🚀 Hướng dẫn chạy chương trình
1. Mở file notebook tại link: [Chèn Link Google Colab của nhóm tại đây].
2. Chọn **Runtime > Run all** (hoặc Runtime > Chạy tất cả).
3. Notebook sẽ tự động:
   - Cài đặt các thư viện cần thiết (`transformers`, `scikit-learn`,...).
   - Tải tập dữ liệu từ nguồn công khai trực tiếp vào môi trường Colab.
   - Thực hiện quy trình EDA, huấn luyện và đánh giá mô hình.

## 📊 Tóm tắt kết quả
*Bảng so sánh chi tiết giữa Machine Learning truyền thống và Deep Learning được trình bày cụ thể trong báo cáo PDF.*
