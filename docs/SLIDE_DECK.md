# SLIDE DECK - BÁO VỆ ĐỒ ÁN

# Hệ thống Gợi ý Sản phẩm Thuê

# CL-GRU4Rec+RP

---

## SLIDE 1: TIÊU ĐỀ

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           HỆ THỐNG GỢI Ý SẢN PHẨM THUÊ                         ║
║     Nâng cao bằng Contrastive Learning - GRU4Rec với              ║
║              Re-Purchase Awareness (CL-GRU4Rec+RP)               ║
║                                                                   ║
═════════════════════════════════════════════════════════════════════
║                                                                   ║
║  GVHD: TS. Nguyễn Văn A                                           ║
║                                                                   ║
║  Nhóm thực hiện:                                                 ║
║    • Thành viên 1: Trần Văn B (Chủ nhiệm)                        ║
║    • Thành viên 2: Lê Thị C                                       ║
║    • Thành viên 3: Phạm Văn D                                     ║
║                                                                   ║
║  Khoa: Công nghệ thông tin                                       ║
║  Trường: Đại học Công nghệ                                       ║
║  Tháng 1/2025                                                     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## SLIDE 2: NỘI DUNG TRÌNH BÀY

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NỘI DUNG TRÌNH BÀY                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Phạm vi bài toán                                               │
│     • Bài toán gốc từ Kaggle Competition                           │
│     • Mở rộng đánh giá đa dataset                                  │
│                                                                     │
│  2. Động lực chọn đề tài                                          │
│     • Bối cảnh kinh tế chia sẻ                                    │
│     • Tiềm năng kinh tế và công nghệ                              │
│     • Thách thức: Data Leakage                                    │
│                                                                     │
│  3. Các công việc liên quan                                       │
│     • Cách tiếp cận truyền thống                                   │
│     • Deep Learning hiện đại                                       │
│     • Khoảng trống nghiên cứu                                      │
│                                                                     │
│  4. Phương pháp đề xuất ⭐                                         │
│     • Tổng quan kiến trúc                                          │
│     • 3 Components chính                                          │
│     • Adaptive Two-Stage Fusion                                    │
│                                                                     │
│  5. Kết quả thực nghiệm                                            │
│     • Thiết lập thí nghiệm                                         │
│     • Kết quả trên Kaggle & Synerise                               │
│     • Ablation Study                                               │
│                                                                     │
│  6. Hướng phát triển                                              │
│     • Explainable AI                                               │
│     • Real-time API Deployment                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 3: 1. PHẠM VI BÀI TOÁN (Bài toán gốc)

```
┌─────────────────────────────────────────────────────────────────────┐
│                  KAGGLE RENTAL COMPETITION                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📋 Giới hạn bài toán:                                             │
│     • Dự đoán sản phẩm thuê tiếp theo của người dùng               │
│     • Dựa trên lịch sử tương tác                                    │
│     • Output: Top-6 sản phẩm gợi ý                                 │
│                                                                     │
│  📊 Đặc điểm dữ liệu:                                              │
│     • ~50,000 người dùng                                           │
│     • ~10,000 sản phẩm                                             │
│     • ~2,000,000 interactions                                      │
│     • 6 tháng dữ liệu                                              │
│                                                                     │
│  🔢 Input:                                                         │
│     user_id, history = [(item₁, event₁, t₁), ...]                  │
│                                                                     │
│  📤 Output:                                                        │
│     [rec₁, rec₂, rec₃, rec₄, rec₅, rec₆]                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 4: 1. PHẠM VI BÀI TOÁN (Mở rộng)

```
┌─────────────────────────────────────────────────────────────────────┐
│               MỞ RỘNG ĐÁNH GIÁ - CROSS DATASET                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  🎯 Mục tiêu mở rộng:                                              │
│     Chứng minh tính khái quát (generalization)                     │
│     của phương pháp trên nhiều datasets khác nhau                   │
│                                                                     │
│  📊 Dataset được đánh giá:                                         │
│                                                                     │
│  ┌─────────────────────┬───────────────────┬───────────────────┐  │
│  │                     │  Kaggle Rental    │  Synerise RecSys  │  │
│  ├─────────────────────┼───────────────────┼───────────────────┤  │
│  │ Ngành hàng          │  Cho thuê         │  E-commerce       │  │
│  │ Users               │  ~50K             │  ~150K            │  │
│  │ Items               │  ~10K             │  ~5K              │  │
│  │ Interactions        │  ~2M              │  ~3.5M            │  │
│  │ Events              │  view,cart,buy    │  view,cart,buy    │  │
│  │ Split method        │  Time-based       │  Per-user 80/20   │  │
│  └─────────────────────┴───────────────────┴───────────────────┘  │
│                                                                     │
│  ✅ Kết quả: Phương pháp hiệu quả trên cả hai datasets             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 5: 2. ĐỘNG LỰC CHỌN ĐỀ TÀI

```
┌─────────────────────────────────────────────────────────────────────┐
│                 KINH TẾ CHIA SẺ (SHARING ECONOMY)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📈 Tăng trưởng:                                                   │
│     CAGR 19.4% (2023-2030)                                         │
│                                                                     │
│  💰 Các nền tảng nổi tiếng:                                        │
│     • Airbnb: nghỉ dưỡng                                            │
│     • Turo: xe cộ                                                  │
│     • Rent the Runway: thời trang                                  │
│     • Fat Llama: đồ điện tử                                        │
│                                                                     │
│  ⚠️ Vấn đề:                                                        │
│     Các hệ thống gợi ý hiện tại được thiết kế                      │
│     cho mua sắm truyền thống → KHÔNG TỐI ƯU cho thuê!             │
│                                                                     │
│  💡 Cơ hội:                                                        │
│     Nghiên cứu RecSys dành riêng cho bài toán cho thuê             │
│     là một khoảng trống nghiên cứu                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 6: 2. TIỀM NĂNG CỦA ĐỀ TÀI

```
┌─────────────────────────────────────────────────────────────────────┐
│              TIỀM NĂNG KINH TẾ & CÔNG NGHỆ                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  💰 Tiềm năng KINH TẾ:                                            │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ • Tăng Conversion Rate (CR)                              │    │
│     │ • Tăng Revenue per User                                 │    │
│     │ • Giảm Churn Rate                                        │    │
│     │ • Phát hiện chu kỳ thuê → Gợi ý nhắc nhu cầu            │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                     │
│  🔬 Tiềm năng CÔNG NGHỆ:                                           │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ • Giải quyết sparse data (dữ liệu thuê thường thưa)    │    │
│     │ • Sequence modeling với mixed signals                    │    │
│     │ • Multi-behavior fusion (view ≠ cart ≠ buy)             │    │
│     │ • Novel approach cho rental domain                       │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 7: 2. THÁCH THỨC: DATA LEAKAGE

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA LEAKAGE PROBLEM                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ⚠️  VẤN ĐỀ PHÁT HIỆN:                                            │
│     Phát hiện sự trùng lặp ID giữa tập train và test               │
│     → Khả năng suy luận nhãn offline (label leakage)               │
│                                                                     │
│  🔍 PHÂN TÍCH:                                                      │
│     ┌──────────────────┐      ┌──────────────────┐                │
│     │   TRAIN SET      │      │    TEST SET      │                │
│     │  Users: A,B,C    │      │  Users: A,B,D    │                │
│     └──────────┬───────┘      └───────┬──────────┘                │
│                │                      │                           │
│                └──────────┬───────────┘                           │
│                           │                                       │
│                    ┌───────▼────────┐                              │
│                    │   OVERLAP:     │                              │
│                    │  A, B (RISK!)  │                              │
│                    └────────────────┘                              │
│                                                                     │
│  ✅ GIẢI PHÁP:                                                      │
│     Implement Per-user 80/20 split thay vì random split            │
│     → Evaluation "sạch" cho mỗi user                               │
│                                                                     │
│  📚 BÀI HỌC: EDA kỹ lưỡng là BẮT BUỘC trước khi train!           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 8: 3. CÁC CÔNG TRÌNH LIÊN QUAN (Truyền thống)

```
┌─────────────────────────────────────────────────────────────────────┐
│           CÁCH TIẾP CẬN TRUYỀN THỐNG                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  1. Collaborative Filtering (CF)                          │    │
│  │  ──────────────────────────────────────────────────────────│    │
│  │  ✅ Ưu điểm: Đơn giản, dễ implement, giải thích được      │    │
│  │  ❌ Nhược điểm:                                            │    │
│  │     • Cold-start problem                                   │    │
│  │     • Sparsity problem                                     │    │
│  │     • Không capture temporal patterns                       │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  2. Matrix Factorization (MF)                             │    │
│  │  ──────────────────────────────────────────────────────────│    │
│  │  ✅ Ưu điểm: Efficient, scalable                          │    │
│  │  ❌ Nhược điểm:                                            │    │
│  │     • Không capture sequential dependencies                │    │
│  │     • Khó xử lý implicit feedback                          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 9: 3. CÁC CÔNG TRÌNH LIÊN QUAN (Deep Learning)

```
┌─────────────────────────────────────────────────────────────────────┐
│              DEEP LEARNING HIỆN ĐẠI CHO RECSYS                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📚 State-of-the-art Models:                                       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • SASRec (2018): Self-Attentive Sequential Recommendation │   │
│  │    → Better long-term dependencies                           │   │
│  │                                                              │   │
│  │  • BERT4Rec (2019): Bidirectional self-attention            │   │
│  │    → State-of-the-art trên nhiều benchmarks                 │   │
│  │                                                              │   │
│  │  • GRU4Rec (2016): First RNN for SBR                        │   │
│  │    → BPR và TOP1 loss functions                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ⚠️  HẠN CHẾ CHUNG:                                                │
│     Coi mỗi interaction độc lập → Bỏ qua re-purchase signals       │
│     Không tận dụng được multi-behavior nature (view ≠ cart ≠ buy)   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 10: 3. KHOẢNG TRỐNG NGHIÊN CỨU

```
┌─────────────────────────────────────────────────────────────────────┐
│                  RESEARCH GAPS IDENTIFIED                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────┬─────────────────────┬────────────────┐  │
│  │ Vấn đề               │ Hiện tại           │ Khoảng trống   │  │
│  ├───────────────────────┼─────────────────────┼────────────────┤  │
│  │ Re-purchase behavior │ Ít được nghiên cứu │ Explicit       │  │
│  │                       │                     │ modeling       │  │
│  ├───────────────────────┼─────────────────────┼────────────────┤  │
│  │ Multi-behavior        │ Treat equally       │ Event-weighted │  │
│  │ fusion                │                     │ approach       │  │
│  ├───────────────────────┼─────────────────────┼────────────────┤  │
│  │ Sequential + Graph    │ Thường tách rời     │ Integrated     │  │
│  │                       │                     │ approach       │  │
│  ├───────────────────────┼─────────────────────┼────────────────┤  │
│  │ Rental domain         │ Không có research   │ Domain-specific│  │
│  │                       │                     │ methods        │  │
│  └───────────────────────┴─────────────────────┴────────────────┘  │
│                                                                     │
│  💡 GIẢI PHÁP CỦA CHÚNG TÔI: CL-GRU4Rec+RP                       │
│     → Đánh straight vào các gaps này                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 11: 4. PHƯƠNG PHÁP ĐỀ XUẤT ⭐ (Tổng quan)

```
╔════════════════════════════════════════════════════════════════════╗
║            CL-GRU4REC+RP: TỔNG QUAN KIẾN TRÚC                    ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║   ┌────────────────────────────────────────────────────────┐      ║
║   │                     INPUT                              │      ║
║   │  User History: [(i₁, e₁), (i₂, e₂), ..., (iₙ, eₙ)]   │      ║
║   └────────────────────────────────────────────────────────┘      ║
║                          │                                        ║
║          ┌───────────────┼───────────────┐                      ║
║          ▼               ▼               ▼                      ║
║   ┌──────────┐    ┌──────────┐    ┌──────────┐                 ║
║   │  GRU4Rec │    │    CL    │    │    RP    │                 ║
║   │  + BPR   │    │          │    │ Awareness│                 ║
║   │          │    │          │    │          │                 ║
║   │ • Seq    │    │ • Item   │    │ • Event- │                 ║
║   │   patterns│    │   seman- │    │   weighted│                 ║
║   │ • Hidden │    │   tics   │    │ • Recency│                 ║
║   └────┬─────┘    └────┬─────┘    └────┬─────┘                 ║
║        │               │               │                         ║
║        └───────────────┴───────────────┘                         ║
║                        │                                         ║
║                        ▼                                         ║
║            ┌──────────────────────────┐                          ║
║            │  ADAPTIVE TWO-STAGE     │                          ║
║            │  FUSION                  │                          ║
║            │  • Stage 1: RP dominant  │                          ║
║            │  • Stage 2: Discovery    │                          ║
║            └──────────────────────────┘                          ║
║                        │                                         ║
║                        ▼                                         ║
║               [rec₁, rec₂, ..., rec_K]                           ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## SLIDE 12: 4. COMPONENT 1: GRU4Rec + BPR Loss

```
┌─────────────────────────────────────────────────────────────────────┐
│                  GRU4REC VỚI BPR LOSS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  🧮 Công thức BPR Loss:                                            │
│     $$                                                           │
│     \mathcal{L}_{BPR} = -\sum_{(u,i,j) \in \mathcal{D}} \ln      │
│     \sigma(\hat{y}_{ui} - \hat{y}_{uj})                           │
│     $$                                                           │
│     Trong đó:                                                      │
│     • $\hat{y}_{ui}$: Score cho positive item                     │
│     • $\hat{y}_{uj}$: Score cho negative sample                   │
│     • $\sigma$: Sigmoid function                                  │
│                                                                     │
│  ✅ Tại sao BPR?                                                  │
│     • Direct ranking optimization                                 │
│     • Efficient với in-batch negative sampling                     │
│     • Proven effectiveness trong RecSys                            │
│                                                                     │
│  📊 Configuration:                                                │
│     • embed_dim: 128                                              │
│     • hidden_dim: 200                                             │
│     • dropout: 0.15                                               │
│     • Ensemble: 3 models (seeds: 42, 123, 456)                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 13: 4. COMPONENT 2: Contrastive Learning

```
┌─────────────────────────────────────────────────────────────────────┐
│            CONTRASTIVE LEARNING CHO ITEM SEMANTICS                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  🎯 Mục tiêu: Learn semantic item embeddings                      │
│                                                                     │
│  🧮 InfoNCE Loss:                                                 │
│     $$                                                           │
│     \mathcal{L}_{CL} = -\ln \frac{\exp(sim(z_a,z_p)/\tau)}      │
│     {\exp(sim(z_a,z_p)/\tau) + \sum \exp(sim(z_a,z_n)/\tau)}   │
│     $$                                                           │
│                                                                     │
│  🔄 Positive Pairs Construction:                                  │
│     Items trong cùng session → positive pairs                      │
│                                                                     │
│  📊 Configuration:                                                │
│     • embed_dim: 64                                               │
│     • temperature: 0.07                                           │
│     • negatives: 256 per batch                                    │
│                                                                     │
│  ✅ Lợi ích:                                                     │
│     • Help với cold-start (items mới có thể được mapped)          │
│     • Enable discovery của related items                          │
│     • Semantic similarity khác collaborative similarity           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 14: 4. COMPONENT 3: Re-Purchase Awareness

````
┌─────────────────────────────────────────────────────────────────────┐
│              RE-PURCHASE AWARENESS SCORING                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  💡 Ý tưởng: Trong bối cảnh rental, người dùng THUÊ LẠI!         │
│                                                                     │
│  🧮 Scoring Function:                                             │
│     ```python                                                     │
│     for item, event in history:                                   │
│         # Event weighting                                         │
│         if event == "buy":      weight = 5.0                      │
│         elif event == "cart":    weight = 2.0                      │
│         else:                   weight = 1.0                      │
│                                                                     │
│         # Recency boost                                           │
│         recency = 1.0 + (position / len(history))                 │
│                                                                     │
│         rp_scores[item] += weight * recency                        │
│     ```                                                           │
│                                                                     │
│  ✅ Tại sao hiệu quả?                                             │
│     • Buy events quan trọng hơn cart/view                         │
│     • Recent items có higher probability được thuê lại             │
│     • Frequency counting captures repeat patterns                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
````

---

## SLIDE 15: 4. ADAPTIVE TWO-STAGE FUSION

```
┌─────────────────────────────────────────────────────────────────────┐
│               ADAPTIVE TWO-STAGE FUSION STRATEGY                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  STAGE 1: Re-Purchase Dominant                                     │
│  ─────────────────────────────────────                             │
│     IF strong RP signal (≥K unique items):                         │
│         → RP fills most slots                                      │
│     ELSE:                                                         │
│         → RP + GRU + CL fill proportionally                         │
│                                                                     │
│  ─────────────────────────────────────                             │
│  STAGE 2: Discovery Fills Remaining Slots                          │
│  ─────────────────────────────────────                             │
│     IF still < K recommendations:                                  │
│         → CL + GRU + CoOccur fill discovery slots                   │
│                                                                     │
│  🔑 Session-Adaptive Weights:                                     │
│     ┌───────────────────┬──────────────────────────────┐           │
│     │ Session Length    │ Fusion Weights              │           │
│     ├───────────────────┼──────────────────────────────┤           │
│     │ ≥ 10 items        │ RP: 80%, Discovery: 20%     │           │
│     │ 3-9 items         │ RP: 50%, Discovery: 50%     │           │
│     │ < 3 items         │ RP: 20%, Discovery: 80%     │           │
│     └───────────────────┴──────────────────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 16: 4. ĐIỂM MỚI & CẢI TIẾN

```
┌─────────────────────────────────────────────────────────────────────┐
│                  KEY INNOVATIONS & CONTRIBUTIONS                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  🎯 INNOVATION 1: Re-Purchase Modeling                            │
│     ─────────────────────────────────────                           │
│     • First RecSys method explicitly designed for rental           │
│     • Event-weighted scoring (buy: 5.0, cart: 2.0, view: 1.0)     │
│     • Recency boost cho recent interactions                        │
│                                                                     │
│  🎯 INNOVATION 2: Separate Training + Inference Fusion             │
│     ─────────────────────────────────────                           │
│     • Avoid multi-task confusion                                  │
│     • Flexible fusion based on user characteristics                │
│     • Easy to debug and extend                                     │
│                                                                     │
│  🎯 INNOVATION 3: Session-Adaptive Fusion                         │
│     ─────────────────────────────────────                           │
│     • Different strategies cho different user types                │
│     • Automatic weight adjustment                                  │
│     • No manual tuning per user segment                            │
│                                                                     │
│  📊 So sánh:                                                        │
│     ┌─────────────────────────────────────────────────────────┐    │
│     │ Mô hình hiện tại  │ CL-GRU4Rec+RP                        │    │
│     ├─────────────────────────────────────────────────────────┤    │
│     │ Ko có RP model    │ ✅ Event-weighted RP scoring        │    │
│     │ Content-based CL   │ ✅ CL từ behavioral data           │    │
│     │ Transformer-based │ ✅ GRU + BPR (efficient)            │    │
│     │ Fixed weights     │ ✅ Adaptive two-stage              │    │
│     └─────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 17: 5. KẾT QUẢ THỰC NGHIỆM

```
┌─────────────────────────────────────────────────────────────────────┐
│              KẾT QUẢ TRÊN SYNERISE RECSYS DATASET (K=10)           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Method              │  R@10  │ NDCG@10 │ HR@10 │ Improvement     │
│   ────────────────────┼────────┼─────────┼───────┼─────────────    │
│   Popularity          │ 0.0345 │ 0.0312  │0.0891 │      -          │
│   RePurchase          │ 0.0823 │ 0.0756  │0.2012 │  +139%          │
│   GRU4Rec             │ 0.1123 │ 0.1056  │0.2789 │  +226%          │
│   CL-GRU4Rec+RP       │ 0.1456 │ 0.1345  │0.3234 │  +322% 🏆       │
│                                                                     │
│   ──────────────────────────────────────────────────────────────    │
│                                                                     │
│   📊 Extended Metrics:                                             │
│      • Novelty:     0.235  (cao nhất → recommend unpopular items)  │
│      • Diversity:   0.712  (tốt → diverse recommendations)        │
│      • Coverage:    0.312  (31% catalog được recommend)           │
│                                                                     │
│   ──────────────────────────────────────────────────────────────    │
│                                                                     │
│   🔍 Key Findings:                                                 │
│      • RP component contributes MOST (+15.2%)                     │
│      • CL adds significant value (+9.1%)                           │
│      • Ensemble training improves stability                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 18: 5. ABLATION STUDY

```
┌─────────────────────────────────────────────────────────────────────┐
│                  ABLATION STUDY (Đóng góp từng component)          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Model Configuration              │  R@10  │   Δ from full        │
│   ─────────────────────────────────┼────────┼──────────────────    │
│   Full CL-GRU4Rec+RP               │ 0.1456 │         -            │
│   ─────────────────────────────────┼────────┼──────────────────    │
│   w/o CL (GRU+RP only)             │ 0.1323 │      -9.1%          │
│   w/o RP (GRU+CL only)             │ 0.1234 │     -15.2% ⚠️       │
│   w/o CoOccurrence                 │ 0.1389 │      -4.6%          │
│   ─────────────────────────────────┼────────┼──────────────────    │
│   GRU4Rec only                     │ 0.1123 │     -22.9%          │
│                                                                     │
│   ──────────────────────────────────────────────────────────────    │
│                                                                     │
│   📊 Visualization:                                                │
│      RP Component    ━━━━━━━━━━━━━━━━━━━━━━ 15.2% ████████████    │
│      CL Component    ━━━━━━━━━━━━━ 9.1%   ███████               │
│      CoOccurrence    ━━━━ 4.6%         ███                      │
│                                                                     │
│      Total improvement over GRU-only: +29.7%                       │
│                                                                     │
│   💡 Conclusion: RP component là QUAN TRỌNG NHẤT                  │
│                nhưng cần kết hợp với CL để đạt hiệu quả tốt nhất   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 19: 6. HƯỚNG PHÁT TRIỂN

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FUTURE WORK & CONCLUSIONS                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  🚀 FUTURE WORK:                                                   │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│  1. Explainable AI (XAI) Integration                              │
│     • Attention-based explanation layers                          │
│     • Template-based explanations ("Because you rented X...")      │
│     • Visual explanation interface                                 │
│                                                                     │
│  2. Real-time API Deployment                                      │
│     • FastAPI + Redis caching                                      │
│     • Target latency: <50ms (p50), <200ms (p99)                    │
│     • Model optimization với TorchScript/ONNX                     │
│                                                                     │
│  3. Seasonal Modeling                                             │
│     • Explicit seasonal embeddings                                 │
│     • Time-aware attention                                        │
│     • Event duration modeling                                      │
│                                                                     │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│  ✅ CONCLUSION:                                                    │
│     • CL-GRU4Rec+RP: Effective method cho rental domain           │
│     • Proven across 2 datasets: +30% over baselines                │
│     • Modular, flexible, production-ready                          │
│     • Opens new research directions cho rental RecSys              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 20: Q&A

```
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                        CẢM ƠN QUÝ THẦY CÔ!                      ║
║                                                                    ║
║                     QUESTIONS & ANSWERS                           ║
║                                                                    ║
║        ______________________________________________________       ║
║       /                                                      \      ║
║      /                                                        \     ║
║     /                                                          \    ║
║    /                                                            \   ║
║   /                                                              \  ║
║  /                                                                \ ║
║ /                                                                  ║
║/                                                                    ║
║\                                                                    ║
║ \                                                                  / ║
║  \                                                                /  ║
║   \                                                              /   ║
║    \                                                            /    ║
║     \                                                          /     ║
║      \                                                        /      ║
║       \                                                      /       ║
║        \____________________________________________________/        ║
║                                                                    ║
║                      THANK YOU FOR LISTENING!                     ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
```
