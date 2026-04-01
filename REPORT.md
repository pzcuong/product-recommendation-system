# Product Recommendation System — Technical Report

## 1. Tổng quan

**Bài toán:** Dự đoán 6 sản phẩm tiếp theo mà người dùng sẽ xem trong một phiên duyệt web (session-based recommendation), đánh giá bằng **Recall@6**.

**Kết quả tốt nhất đạt được:**

| Submission                      | Private Score | Public Score |
| ------------------------------- | ------------- | ------------ |
| GRU4Rec (150) + Re-rank         | 0.40924       | 0.39096      |
| **GRU4Rec × 3 seeds + Re-rank** | **0.41012**   | **0.41650**  |
| Baseline (co-occurrence only)   | 0.38113       | —            |

---

## 2. Phân tích dữ liệu

| Thông số                        | Giá trị           |
| ------------------------------- | ----------------- |
| Tổng số sự kiện                 | 1,295,242         |
| Số người dùng (client_id)       | 188,131           |
| Số phiên duyệt (visit_id)       | 276,162           |
| Số sản phẩm trong catalog       | 1,056 items       |
| Số sản phẩm cho phép (new site) | ~688              |
| Số phiên trong tập test         | 1,363             |
| Cold-start (không có lịch sử)   | 53 / 1,363 (3.9%) |

**Đặc điểm dữ liệu:**

- Mỗi visit chứa chuỗi các sự kiện: `PRODUCT`, `CATEGORY`, `SEARCH`, `CART`, `ORDER`...
- Hai nguồn site: old site → new site (sản phẩm được map ID)
- Dữ liệu rất sparse (~99.8%): 1.7M hits / 237M tổ hợp có thể
- 73% validation sessions là `cold_single_cat` (rất ít lịch sử)

---

## 3. Pipeline tổng thể

```mermaid
flowchart TD
    A[metrika_hits.csv\nmetrika_hits_test.csv] --> C[get_hits_data\nSlug → product_id mapping]
    B[metrika_visits.csv\nmetrika_visits_test.csv] --> D[get_visits_data\nExplode watch_ids]
    E[old_site_products.csv\nnew_site_products.csv\nold_site_new_site_products.csv] --> C

    C --> F[create_recom_data\nMerge hits + visits\nDeduplicate PRODUCT events]
    D --> F

    F --> G[add_start_token\nThêm token START_OLD / START_NEW]

    G --> H{SKIP_VALIDATION?}

    H -- False --> I[get_data_splits 50/50\ndf_train | df_valid | df_test]
    I --> J[get_fitted_model seed=123\nGRU4Rec validation model]
    J --> K[create_submission\nRecall@6 trên valid + test]
    K --> L[📊 Validation recall@6: 0.3118\nTest recall@6: 0.3971]

    H -- True / After validation --> M

    G --> M[Multi-Seed Ensemble Training\nSeeds: 123, 456, 789]
    M --> M1[GRU4Rec seed=123\n150 layers · 30 epochs\nn_sample=4096]
    M --> M2[GRU4Rec seed=456\n150 layers · 30 epochs\nn_sample=4096]
    M --> M3[GRU4Rec seed=789\n150 layers · 30 epochs\nn_sample=4096]
    M1 --> N[Score Averaging\nTop-100 candidates / visit\nAvg score = sum / 3 models]
    M2 --> N
    M3 --> N

    G --> O[Build Co-occurrence Signals\nfrom df_data grouped by client_id]
    O --> O1[forward_cooccur\nA → B nếu B xem sau A]
    O --> O2[backward_cooccur\nB → A nếu A xem trước B]
    O --> O3[cat_to_products\nCategory slug → products]
    O --> O4[visit_context\nLast 5 products + last 3 categories\ncho mỗi test visit]

    N --> P[Prediction-Level Fusion\nRe-rank top-100 với co-occurrence signals]
    O1 --> P
    O2 --> P
    O3 --> P
    O4 --> P

    P --> Q[Session-Adaptive Boost Cap\nn_history ≥ 3 → cap 25%\nn_history = 1-2 → cap 15%\nn_history = 0 → cap 8%]

    Q --> R[Per-item Score Boost\n+ Forward cooccur × recency 1x–3x\n+ Backward cooccur × recency 1x–2.5x\n+ Category signal × recency 1x–2x]

    R --> S[Sort re-ranked top-6\nper visit_id]

    S --> T[Finalize Submission\nCold start: 53 visits\nFill với global popular items]

    T --> U[Reindex theo test visit_id order\nfrom metrika_visits_test.csv]

    U --> V[submission.csv\n1363 rows × 6 products]
```

---

## 4. Các thành phần kỹ thuật

### 4.1 GRU4Rec (Neural Sequential Model)

Dùng thư viện **Cornac 2.3.5** — implementation GRU4Rec gốc của Hidasi et al. (2016).

| Hyperparameter        | Giá trị                           |
| --------------------- | --------------------------------- |
| `layers`              | [150] (1 hidden layer, 150 units) |
| `n_epochs`            | 30                                |
| `loss`                | cross-entropy                     |
| `n_sample`            | 4096 (negative samples)           |
| `batch_size`          | 512                               |
| `dropout_p_embed`     | 0.0                               |
| `dropout_p_hidden`    | 0.0                               |
| `device`              | MPS (Apple Silicon)               |
| `max_row_per_session` | 20                                |
| `split_ratio`         | 0.50 (train/valid)                |

**Float32 patch:** Cornac khởi tạo weights bằng `float64` → không tương thích MPS. Patch `_init_numpy_weights` để ép về `float32`.

**Input format:** `(client_id, visit_id, product_id, timestamp)` → `SequentialDataset.build(fmt="USIT")`.

**Dự đoán:** Với mỗi visit, lấy toàn bộ lịch sử sản phẩm → GRU hidden state → score vector trên toàn bộ catalog → argpartition top-k.

### 4.2 Multi-Seed Ensemble

Train 3 model độc lập với seed khác nhau (123, 456, 789), sau đó **average scores** tại prediction time:

```
score_avg(item, visit) = Σ score_i(item, visit) / 3
```

Items không xuất hiện trong top-100 của một model → score = 0 trong model đó.

Kết quả: giảm variance của từng model → tăng nhẹ recall.

### 4.3 Co-occurrence Signals

Build từ toàn bộ `df_data` (train + test history), group by `client_id`:

- **Forward co-occurrence** `forward_cooccur[A][B]`: số lần B được xem ngay sau A trong cùng session (window = 5).
- **Backward co-occurrence** `backward_cooccur[B][A]`: chiều ngược lại (lighter weight).
- **Category → Products** `cat_to_products[cat][pid]`: số lần sản phẩm pid được xem sau khi browse category `cat` (window = 3 categories).

### 4.4 Prediction-Level Fusion (Re-ranking)

**Thiết kế nguyên tắc:** GRU4Rec là primary model. Co-occurrence chỉ được phép **re-rank** trong top-100 candidates của GRU4Rec — không inject item từ ngoài (đã thử và làm giảm 0.40924 → 0.39005).

**Boost formula** cho mỗi item trong top-100:

```
boost = Σ forward_boost(prev, item) × recency_fwd
      + Σ backward_boost(prev, item) × recency_bwd
      + Σ category_boost(cat, item) × recency_cat

boost_capped = min(boost, |base_score| × boost_cap_pct)
final_score  = base_score + boost_capped
```

**Recency weighting:**

- Forward: multiplier từ 1.0×(xa) đến 3.0×(gần nhất)
- Backward: multiplier từ 1.0× đến 2.5×
- Category: multiplier từ 1.0× đến 2.0×

**Session-adaptive cap:**

- n_history ≥ 3 items → max boost = **25%** of base score
- n_history = 1–2 → max boost = **15%**
- n_history = 0 (cold start) → max boost = **8%**

---

## 5. Những thử nghiệm đã qua

| Thử nghiệm                                | Kết quả        | Ghi chú                         |
| ----------------------------------------- | -------------- | ------------------------------- |
| Co-occurrence only (baseline)             | 0.381          | Không có GRU4Rec                |
| GRU4Rec (100 layers, 20 epochs)           | 0.409          | Từ kaggle_notebook.py cũ        |
| Score-mixing ensemble (v16+v42+v45)       | **0.296** ❌   | 3 lỗi nghiêm trọng trong fusion |
| GRU4Rec (150, 30) + simple re-rank top-50 | 0.40837        |                                 |
| + backward cooccur + recency + top-100    | 0.40924        |                                 |
| GRU4Rec (200 layers, 50 epochs)           | 0.385          | Overfitting                     |
| + Inject ngoài top-100                    | 0.39005 ❌     | Inject phá GRU4Rec              |
| **Multi-seed (×3) + re-rank**             | **0.41012** ✅ | Bản tốt nhất                    |

### Tại sao score-mixing ensemble thất bại (0.296)?

```
Bug 1: v16 base_bias = 4.0×1.3 = 5.2
       GRU4Rec base_bias = 5.0×0.3 = 1.5
       → v16 items luôn thắng GRU4Rec

Bug 2: Consensus stacking: score *= c2 *= c3 *= c4
       → amplification lên đến 2.53× cho items được nhiều model chọn
       → v16/v42/v45 dùng cùng data nên luôn đồng ý nhau

Bug 3: v16, v42, v45 đều dùng cat_to_products + forward_cooccur
       → fake consensus, không phải 4 model thực sự độc lập
```

---

## 6. Cold Start

53/1363 test visits (3.9%) hoàn toàn không có lịch sử xem sản phẩm hay category. Tất cả đều được fill bằng **global popular items** (top-6 sản phẩm xuất hiện nhiều nhất trong predictions của warm sessions).

---

## 7. Kết luận và giới hạn

**Ceiling thực tế với dataset này:**

- GRU4Rec đã capture gần hết sequential patterns trong data
- 99.8% sparsity giới hạn co-occurrence signals (chỉ 588 items có co-occurrence đủ mạnh)
- 3 seed × GRU4Rec = ~0.001 gain — diminishing returns từ ensemble nhanh

**Để cải thiện vượt 0.42 cần:**

1. External product features (ảnh, text description) → content-based filtering
2. User demographic signals
3. Dữ liệu order/checkout để học purchase patterns
4. Session-level transformer (BERT4Rec, SASRec) thay GRU4Rec

---

_Ngày: March 2026 | Dataset: Baby product rental marketplace | Metric: Recall@6_
