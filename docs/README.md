# Tài liệu đồ án Hệ thống Gợi ý Sản phẩm Thuê

## Thư mục này chứa các tài liệu liên quan đến đồ án:

### 📄 Báo cáo chi tiết

- **`BAO_CAO_DO_AN_CL_GRU4REC_RP.md`**: Báo cáo khoa học đầy đủ (30-50 trang khi in)
  - 6 chương chính
  - Công thức toán học chi tiết
  - Code examples minh họa
  - Kết quả thực nghiệm

### 📊 Slide trình bày

- **`SLIDE_DECK.md`**: Nội dung slide cho bảo vệ đồ án
  - 19 slides theo cấu trúc chuẩn
  - Trực quan hóa kiến trúc
  - Bảng kết quả so sánh

### 📐 Công thức toán học

Các công thức trong báo cáo được viết bằng LaTeX:

$$
\mathcal{L}_{BPR} = -\sum_{(u,i,j) \in \mathcal{D}} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
$$

### 🛠️ Công cụ xem tài liệu

**Online:**

- GitHub: Xem trực tiếp (rendering tốt cho markdown và LaTeX)
- GitLab: Tương tự GitHub

**Offline:**

```bash
# Cài đặt Pandoc để convert sang PDF
brew install pandoc
brew install basictex  # macOS

# Convert sang PDF
pandoc BAO_CAO_DO_AN_CL_GRU4REC_RP.md -o bao_cao.pdf \
  --pdf-engine=xelatex \
  --template=eisvogel \
  --toc --toc-depth=3 \
  --highlight-style=tango \
  --resource-path=.
```

**VS Code:**

- Cài extension: Markdown Preview Enhanced
- Mở file và chọn "Open Preview to the Side"

### 📚 Cấu trúc báo cáo

```
BAO_CAO_DO_AN_CL_GRU4REC_RP.md
├── Lời mở đầu
├── Mục lục
├── Chương 1: Tổng quan
│   ├── 1.1 Giới thiệu hệ thống gợi ý
│   ├── 1.2 Hệ thống gợi ý cho bài toán cho thuê
│   └── 1.3 Đặt vấn đề
├── Chương 2: Cơ sở lý thuyết
│   ├── 2.1 Collaborative Filtering
│   ├── 2.2 Matrix Factorization
│   ├── 2.3 Deep Learning cho RecSys
│   ├── 2.4 Session-Based Recommendation
│   └── 2.5 Contrastive Learning
├── Chương 3: Các công trình liên quan
│   ├── 3.1 Phương pháp truyền thống
│   ├── 3.2 Deep Learning hiện đại
│   └── 3.3 Khoảng trống nghiên cứu
├── Chương 4: Phương pháp đề xuất (TRỌNG TÂM)
│   ├── 4.1 Tổng quan kiến trúc
│   ├── 4.2 Component 1: GRU4Rec với BPR Loss
│   ├── 4.3 Component 2: Contrastive Learning
│   ├── 4.4 Component 3: Re-Purchase Awareness
│   ├── 4.5 Adaptive Two-Stage Fusion
│   └── 4.6 Điểm mới và cải tiến
├── Chương 5: Triển khai và thực nghiệm
│   ├── 5.1 Mô tả dữ liệu
│   ├── 5.2 Thiết lập thí nghiệm
│   ├── 5.3 Kết quả Kaggle Rental
│   ├── 5.4 Kết quả Synerise RecSys
│   ├── 5.5 Ablation Study
│   └── 5.6 Phân tích trường hợp thất bại
├── Chương 6: Hướng phát triển
│   ├── 6.1 Explainable AI
│   ├── 6.2 Real-time API
│   └── 6.3 Seasonal Modeling
├── Kết luận
├── Tài liệu tham khảo
└── Phụ lục
```

### 🔗 Liên kết với code chính

Báo cáo tham chiếu đến các file implementation:

- `../cl_gru4rec_rp_unified.py` - Main model
- `../cl_gru4rec_rp_v3.py` - BPR variant
- `../cl_gru4rec_rp_academic.py` - Academic evaluation

### 📝 Ghi chú

- Tài liệu được viết bằng Tiếng Việt
- Sử dụng định dạng Markdown với LaTeX cho công thức
- Độ dài ~2000 dòng (khoảng 40-50 trang A4 khi in)
- Tương thích với các markdown viewer phổ biến
