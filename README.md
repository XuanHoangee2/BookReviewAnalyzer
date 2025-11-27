# BookReviewAnalyzer
📘 Tổng quan dự án
1. Giới thiệu
Dự án tập trung fine-tuning mô hình "unsloth/Meta-Llama-3.1-8B-Instruct" bằng phương pháp QLoRA, sử dụng bộ dữ liệu đánh giá sách Amazon Reviews 2023("cogsci13/Amazon-Reviews-2023-Books-Review"). Mục tiêu là xây dựng một mô hình có khả năng phân tích đánh giá sách của người dùng.

2. Dữ liệu
Bộ dữ liệu được tải từ HuggingFace, sau đó được làm sạch, loại bỏ nhiễu, chuẩn hóa văn bản giúp mô hình học tốt hơn các nhiệm vụ như phân tích cảm xúc và tóm tắt.

3. Phương pháp
QLoRA được sử dụng để giảm kích thước mô hình xuống 4-bit, cho phép huấn luyện mô hình 8B trên GPU giới hạn nhưng vẫn giữ được chất lượng tham số và khả năng học. Phương pháp này tiết kiệm tài nguyên nhưng vẫn đảm bảo hiệu quả.

4. Kết quả
Sau huấn luyện, mô hình cho thấy khả năng cải thiện rõ rệt trong việc nhận diện cảm xúc, tóm tắt nội dung, và phản hồi tự nhiên dựa trên các đánh giá sách. Mô hình ổn định hơn với dữ liệu dài và phản hồi có tính mạch lạc cao.

5. Ứng dụng
Mô hình có thể được sử dụng trong các hệ thống gợi ý sách, phân loại phản hồi khách hàng, chatbot hỗ trợ phân tích review, hoặc các nhiệm vụ NLP liên quan đến đánh giá sản phẩm.
