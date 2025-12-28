# IP History Strategy: Giải pháp "Cold Start" Đột phá (Score 0.74)

## 1. Tổng quan (Introduction)
Dự án này tập trung vào việc tối ưu hóa hệ thống gợi ý sản phẩm (Recommendation System) cho một sàn thương mại điện tử. Thách thức lớn nhất là vấn đề **"Cold Start"** (Khởi động lạnh): Làm thế nào để gợi ý chính xác cho người dùng mới (New/Anonymous Users) khi chưa có lịch sử hành vi?

Giải pháp truyền thống (Global Popularity) chỉ đạt Recall@6 khoảng **0.45**, kéo lùi điểm số tổng thể xuống **0.62**.

Chúng tôi đã phát triển phương pháp **"IP-Based History Transfer"**, giúp tận dụng lịch sử của các thiết bị khác trong cùng hộ gia đình/mạng lưới để "làm ấm" (hydrate) các session lạnh. Kết quả là Recall@6 của nhóm Cold User tăng vọt lên **0.68**, và điểm tổng thể đạt **0.74**.

---

## 2. Dữ liệu & Insight (Data & Discovery)

### Dữ liệu đầu vào
- **`metrika_hits.csv`**: Chứa hơn 1.7 triệu lượt xem sản phẩm.
- **`metrika_visits.csv`**: Thông tin session, bao gồm `ip_address`, `client_id` (cookie), `user_hash`.
- **`metrika_hits_test.csv`**: Tập dữ liệu cần dự đoán (Test Set).

### Insight Quan trọng (The Breakthrough Discovery)
Khi phân tích sự trùng lặp (Overlap) giữa tập huấn luyện (Train) và tập kiểm tra (Test), chúng tôi phát hiện ra một điều thú vị:

- **User ID Overlap:** 0% (Hiển nhiên, vì kịch bản là Cold Start).
- **Client ID Overlap:** 1.2% (Rất thấp, do xóa cookie/đổi trình duyệt).
- **IP Address Overlap:** **99.6%** (Gần như tuyệt đối).

**Kết luận:** Hầu hết "User lạ" trong tập Test thực chất là những người dùng đã từng xuất hiện trong quá khứ, nhưng họ sử dụng thiết bị mới hoặc xóa cookie. Tuy nhiên, họ vẫn truy cập từ cùng một địa chỉ IP (Nhà riêng, Wifi công ty).

=> **Ý tưởng:** Nếu không biết User là ai, hãy xem IP của họ đã từng quan tâm gì.

---

## 3. Phương pháp Tiếp cận (Methodology)

### Lý do chọn (Rationale)
- **Coverage:** IP Address bao phủ 99.6% tập Test, vượt xa mọi signals khác (Search Query, Cart, Order, ClientID).
- **Tính ổn định:** IP hộ gia đình thường không đổi trong session ngắn hạn.
- **Tâm lý hành vi:** Các thành viên trong cùng một hộ gia đình (hoặc chính user đó trên thiết bị khác) thường có xu hướng quan tâm đến các nhóm sản phẩm tương tự nhau.

### Pipeline Xử lý (Implementation)

1.  **Xây dựng IP History (`build_ip_history.py`):**
    - Quét toàn bộ lịch sử xem (`metrika_hits.csv`).
    - Gom nhóm (Group by) theo `ip_address`.
    - Lưu danh sách 10-20 sản phẩm được xem gần nhất tại mỗi IP vào `src/ip_history.pkl`.

2.  **Tích hợp vào Engine (`generate_submission.py`):**
    Hệ thống V7.1 sử dụng chiến thuật "Tràn tầng" (Cascading Fallback):
    
    *   **Tầng 1 (User History):** Nếu nhận diện được User Hash -> Dùng lịch sử cá nhân (Độ chính xác cao nhất).
    *   **Tầng 2 (IP History - MỚI):** Nếu User lạ, kiểm tra IP Address. Nếu IP này có lịch sử -> Gợi ý sản phẩm từ lịch sử IP.
    *   **Tầng 3 (Global Popularity):** Nếu IP cũng lạ -> Dùng sản phẩm phổ biến toàn sàn.

3.  **Mô hình (Model):**
    - Sử dụng **LightGBM Reranker** ở tầng cuối cùng để sắp xếp lại các ứng viên từ IP History (nếu có candidate retrieval phức tạp hơn), hoặc đơn giản là lấy Top-N recency từ IP history (hiện tại đang dùng Recency thuần túy cho IP).

---

## 4. Input & Output

### Input
Một session cần được gợi ý, ví dụ:
```json
{
    "visit_id": "123456",
    "user_hash": null,  // Không xác định (Cold)
    "curr_item": "5566",
    "ip_address": "192.168.1.5"
}
```

### Process
1.  Hệ thống tra cứu `user_hash` -> Không thấy.
2.  Hệ thống tra cứu `ip_address` ("192.168.1.5") trong `ip_history.pkl`.
3.  Tìm thấy: IP này từng xem `[8899, 7744, 1122]` vào hôm qua.

### Output
```json
{
    "visit_id": "123456",
    "recommendations": [8899, 7744, 1122, 5566, ...] // 5566 là item hiện tại
}
```

---

## 5. Kết quả (Results)

Chúng tôi đã kiểm chứng phương pháp này trên tập Validation cục bộ (mô phỏng đúng phân phối Kaggle):

| Metric | Chiến thuật Cũ (V7.0) | Chiến thuật Mới (V7.1 IP) | Cải thiện |
| :--- | :--- | :--- | :--- |
| **Warm Recall@6** | 0.8011 | 0.8011 | 0% |
| **Cold Recall@6** | 0.4531 | **0.6785** | **+49.7%** |
| **Blended Score**| 0.6271 | **0.7398** | **+18.0%** |

### Kết luận
Chiến thuật **IP History** đã chứng minh hiệu quả vượt trội trong việc giải quyết bài toán Cold Start, giúp hệ thống đạt ngưỡng điểm **0.74**, mức điểm thuộc nhóm dẫn đầu (Top Tier).
