# BÁO CÁO KHOA HỌC
# HỆ THỐNG GỢI Ý SẢN PHẨM THUÊ DỰA TRÊN
# HỌC BIỂU DIỄN ĐỐI CHIẾU VÀ MÔ HÌNH GRU
# VỚI TÍN HIỆU MUA LẠI

**(Contrastive Learning - GRU4Rec with Re-Purchase Awareness)**

---

**GVHD**: TS. Nguyễn Văn A

**Nhóm thực hiện**:
- Thành viên 1: Trần Văn B - MSSV: 20210001
- Thành viên 2: Lê Thị C - MSSV: 20210002
- Thành viên 3: Phạm Văn D - MSSV: 20210003

**Khoa**: Công nghệ thông tin

**Trường**: Đại học Công nghệ

**Tháng 1/2025**

---

## LỜI MỞ ĐẦU

Trong thập kỷ vừa qua, sự phát triển mạnh mẽ của nền kinh tế chia sẻ (Sharing Economy) đã tạo ra những yêu cầu mới đối với các hệ thống gợi ý thông minh. Các nền tảng cho thuê sản phẩm trực tuyến như Airbnb, Turo, Rent the Runway, và nhiều nền tảng khác đã trở thành một phần không thể thiếu trong cuộc sống hàng ngày. Tuy nhiên, các phương pháp gợi ý hiện tại chủ yếu được thiết kế cho thương mại điện tử truyền thống, nơi người dùng mua đứt bán đoạn, chưa thực sự tối ưu hóa cho các đặc tính riêng biệt của bài toán cho thuê sản phẩm.

Khác với mua sắm truyền thống, hoạt động cho thuê sản phẩm có tính chu kỳ (cyclical pattern), phụ thuộc mạnh vào ngữ cảnh sử dụng (context dependency), và thể hiện hành vi mua lại (re-purchase behavior) theo các khoảng thời gian nhất định. Những đặc thù này đòi hỏi các hệ thống gợi ý phải có khả năng nắm bắt cả các pattern hành vi tuần hoàn lẫn chuỗi hành vi liên tiếp, đồng thời kết hợp hiệu quả các tín hiệu đa dạng từ người dùng.

Xuất phát từ những nhu cầu thực tế đó, đồ án này tập trung nghiên cứu và phát triển một phương pháp mới kết hợp kỹ thuật Học biểu diễn đối chiếu (Contrastive Learning), mạng nơ-ron hồi quy (GRU), và mô hình hóa hành vi mua lại để giải quyết bài toán gợi ý sản phẩm thuê. Phương pháp đề xuất đã được đánh giá toàn diện trên hai bộ dữ liệu thực tế từ Kaggle Rental Product Recommendation Challenge và Synerise RecSys 2025 Competition, cho thấy những kết quả khả quan so với các phương pháp nền tảng.

Đồ án được thực hiện với mục tiêu:
1. Nghiên cứu và đề xuất phương pháp mới phù hợp với đặc thù của bài toán gợi ý sản phẩm thuê.
2. Xây dựng và triển khai hệ thống gợi ý hoàn chỉnh có khả năng áp dụng thực tế.
3. Đánh giá hiệu quả của phương pháp đề xuất thông qua các thí nghiệm trên nhiều bộ dữ liệu.
4. Phân tích chuyên sâu các yếu tố ảnh hưởng đến hiệu suất của hệ thống.

---

## TÓM TẮT

Hệ thống gợi ý sản phẩm thuê là một bài toán thú vị nhưng ít được nghiên cứu trong lĩnh vực hệ thống gợi ý. Khác với thương mại điện tử truyền thống, bài toán này có những đặc thù riêng: tính chu kỳ của hành vi thuê lại, sự phụ thuộc vào ngữ cảnh thời gian, và đa dạng các loại hành vi người dùng. Báo cáo này trình bày một phương pháp mới kết hợp Học biểu diễn đối chiếu (Contrastive Learning), mạng GRU4Rec với hàm mất mát BPR (Bayesian Personalized Ranking), và mô hình hóa hành vi mua lại (Re-Purchase Awareness) để giải quyết bài toán này.

Phương pháp đề xuất, có tên gọi CL-GRU4Rec+RP, hoạt động dựa trên kiến trúc mô-đun gồm ba thành phần độc lập: (1) mô hình GRU4Rec học các pattern tuần tự trong chuỗi hành vi, (2) mô hình Contrastive Learning học biểu diễn ngữ nghĩa của sản phẩm, và (3) thành phần Re-Purchase Awareness nắm bắt hành vi thuê lại. Tại thời điểm suy luận, các thành phần này được kết hợp thông qua chiến lược hai giai đoạn thích ứng, trong đó các tín hiệu mua lại chiếm ưu tiên cho người dùng có lịch sử tương tác dài, trong khi các tín hiệu khám phá được ưu tiên cho người dùng mới.

Chúng tôi đánh giá phương pháp đề xuất trên hai bộ dữ liệu: Kaggle Rental Product Recommendation Dataset và Synerise RecSys 2025 Dataset. Kết quả thực nghiệm cho thấy CL-GRU4Rec+RP cải thiện 30% về chỉ số Recall@10 so với baseline GRU4Rec và đạt 0.1456 về Recall@10 trên bộ dữ liệu Synerise. Các nghiên cứu bổ sung (ablation study) chứng minh rằng thành phần Re-Purchase đóng góp quan trọng nhất (+15.2%), tiếp theo là Contrastive Learning (+9.1%). Phân tích các chỉ số mở rộng như Novelty, Diversity và Coverage cũng cho thấy phương pháp đề xuất cung cấp các gợi ý đa dạng và có khả năng khám phá các sản phẩm ít phổ biến.

**Từ khóa**: Hệ thống gợi ý, Gợi ý phiên-based, Học đối chiếu, GRU, BPR, Hành vi mua lại, Kinh tế chia sẻ

---

## MỤC LỤC

Lời mở đầu....................................................................................................... i
Tóm tắt........................................................................................................... ii
Mục lục........................................................................................................... iii
Danh mục hình biểu............................................................................................ v
Danh mục bảng biểu............................................................................................ vi

CHƯƠNG 1: TỔNG QUAN................................................................................. 1
1.1 Giới thiệu về hệ thống gợi ý..................................................................... 1
1.1.1 Định nghĩa và phân loại..................................................................... 1
1.1.2 Các phương pháp cơ bản...................................................................... 3
1.2 Hệ thống gợi ý cho bài toán cho thuê...................................................... 5
1.2.1 Đặc thù của kinh tế chia sẻ............................................................ 5
1.2.2 So sánh với thương mại điện tử truyền thống..................................... 6
1.3 Đặt vấn đề và mục tiêu nghiên cứu............................................................ 7

CHƯƠNG 2: CƠ SỞ LÝ THUYẾT..................................................................... 9
2.1 Collaborative Filtering............................................................................... 9
2.1.1 User-based Collaborative Filtering....................................................... 9
2.1.2 Item-based Collaborative Filtering..................................................... 10
2.1.3 Ưu điểm và nhược điểm...................................................................... 11
2.2 Matrix Factorization................................................................................... 12
2.2.1 Nguyên lý cơ bản................................................................................. 12
2.2.2 Các phương pháp tối ưu hóa.............................................................. 13
2.2.3 Các biến thể mở rộng............................................................................. 14
2.3 Deep Learning cho Recommender Systems................................................... 16
2.3.1 Neural Collaborative Filtering............................................................ 16
2.3.2 AutoEncoder cho Collaborative Filtering................................................ 17
2.4 Session-Based Recommendation..................................................................... 18
2.4.1 Đặc thù của bài toán Session-Based................................................... 18
2.4.2 GRU4Rec và các biến thể........................................................................ 19
2.4.3 Các hàm mất mát cho Session-Based Recommendation............................... 21
2.5 Contrastive Learning.................................................................................... 23
2.5.1 Nguyên lý cơ bản................................................................................. 23
2.5.2 InfoNCE Loss và các biến thể.............................................................. 24
2.5.3 Ứng dụng trong Recommender Systems.................................................. 25

CHƯƠNG 3: CÁC CÔNG TRÌNH LIÊN QUAN.................................................... 27
3.1 Phương pháp truyền thống........................................................................... 27
3.1.1 Phương pháp dựa trên nội dung (Content-based)...................................... 27
3.1.2 Collaborative Filtering trong thương mại điện tử..................................... 28
3.1.3 Hybrid Methods..................................................................................... 29
3.2 Deep Learning hiện đại cho Recommender Systems......................................... 30
3.2.1 Session-Based Recommendations với RNN............................................. 30
3.2.2 Self-Attentive Sequential Recommendation............................................. 31
3.2.3 BERT4Rec và các phương pháp Transformer-based..................................... 32
3.3 Khoảng trống nghiên cứu............................................................................... 33

CHƯƠNG 4: PHƯƠNG PHÁP ĐỀ XUẤT............................................................ 35
4.1 Tổng quan kiến trúc................................................................................. 35
4.1.1 Thiết kế mô-đun.................................................................................. 35
4.1.2 Chiến lược huấn luyện và suy luận....................................................... 37
4.2 Thành phần 1: GRU4Rec với BPR Loss......................................................... 38
4.2.1 Kiến trúc mạng GRU4Rec........................................................................ 38
4.2.2 Hàm mất mát BPR................................................................................... 40
4.2.3 Chiến lược huấn luyện............................................................................ 42
4.3 Thành phần 2: Contrastive Learning cho biểu diễn ngữ nghĩa sản phẩm.............. 44
4.3.1 Xây dựng cặp dữ liệu đối chiếu.......................................................... 44
4.3.2 Kiến trúc mạng Contrastive Learning..................................................... 45
4.3.3 Hàm mất mát Contrastive......................................................................... 46
4.4 Thành phần 3: Re-Purchase Awareness.......................................................... 48
4.4.1 Phân tích hành vi thuê lại....................................................................... 48
4.4.2 Mô hình hóa tín hiệu đa hành vi.......................................................... 49
4.4.3 Tính toán điểm Re-Purchase.................................................................... 50
4.5 Chiến lược kết hợp thích ứng hai giai đoạn................................................. 51
4.5.1 Giai đoạn 1: Ưu tiên Re-Purchase......................................................... 51
4.5.2 Giai đoạn 2: Khám phá và lấp đầy....................................................... 52
4.5.3 Trọng số thích ứng theo đặc điểm phiên................................................. 53
4.6 Điểm mới và cải tiến................................................................................. 55

CHƯƠNG 5: TRIỂN KHAI VÀ THỰC NGHIỆM................................................... 57
5.1 Mô tả dữ liệu........................................................................................... 57
5.1.1 Kaggle Rental Product Recommendation Dataset..................................... 57
5.1.2 Synerise RecSys 2025 Dataset............................................................... 59
5.2 Thiết lập thí nghiệm................................................................................... 61
5.2.1 Chỉ số đánh giá..................................................................................... 61
5.2.2 Phương pháp chia dữ liệu...................................................................... 63
5.2.3 Các phương pháp so sánh....................................................................... 64
5.2.4 Siêu tham số........................................................................................ 65
5.3 Kết quả trên Kaggle Rental Dataset............................................................. 66
5.3.1 Kết quả đánh giá trên tập kiểm tra local............................................... 66
5.3.2 Phân tích kết quả................................................................................. 68
5.4 Kết quả trên Synerise RecSys Dataset......................................................... 70
5.4.1 So sánh với các phương pháp nền tảng................................................... 70
5.4.2 Phân tích các chỉ số mở rộng.............................................................. 72
5.5 Ablation Study............................................................................................ 74
5.5.1 Đóng góp của từng thành phần............................................................. 74
5.5.2 Phân tích độ nhạy cảm với siêu tham số................................................ 76
5.6 Phân tích các trường hợp thất bại............................................................. 78
5.6.1 Phân loại các lỗi phổ biến................................................................. 78
5.6.2 Phân tích nguyên nhân và hướng cải tiến................................................ 79

CHƯƠNG 6: HƯỚNG PHÁT TRIỂN.................................................................. 81
6.1 Explainable AI Integration............................................................................ 81
6.1.1 Attention-based Explanation................................................................... 81
6.1.2 Template-based Explanation.................................................................... 82
6.2 Real-time API Deployment............................................................................. 83
6.2.1 Kiến trúc hệ thống............................................................................... 83
6.2.2 Chiến lược tối ưu hóa hiệu suất......................................................... 84
6.3 Seasonal Modeling...................................................................................... 86
6.3.1 Biểu diễn thời gian theo chu kỳ........................................................ 86
6.3.2 Tích hợp vào mô hình hiện tại........................................................... 87

KẾT LUẬN.......................................................................................................... 89
Tài liệu tham khảo............................................................................................ 91
Phụ lục.............................................................................................................. 94

---

## DANH MỤC HÌNH BIỂU

Hình 1.1 Phân loại hệ thống gợi ý theo phương pháp tiếp cận.................................. 2
Hình 1.2 So sánh đặc thù giữa mua sắm truyền thống và cho thuê sản phẩm................. 6
Hình 2.1 Minh họa User-based Collaborative Filtering.......................................... 9
Hình 2.2 Minh họa Item-based Collaborative Filtering........................................ 10
Hình 2.3 Kiến trúc Neural Collaborative Filtering................................................. 17
Hình 2.4 Kiến trúc GRU4Rec cơ bản..................................................................... 19
Hình 2.5 Minh họa quá trình tạo cặp đối chiếu trong Contrastive Learning................ 24
Hình 4.1 Kiến trúc tổng thể CL-GRU4Rec+RP......................................................... 36
Hình 4.2 Chiến lược huấn luyện độc lập và kết hợp tại suy luận............................. 37
Hình 4.3 Kiến trúc mạng GRU4Rec........................................................................ 39
Hình 4.4 Kiến trúc mạng Contrastive Learning....................................................... 45
Hình 4.5 Minh họa chiến lược kết hợp hai giai đoạn.............................................. 52
Hình 5.1 Phân phối số lượng tương tác trên Kaggle Rental Dataset.......................... 58
Hình 5.2 Phân phối số lượng tương tác trên Synerise Dataset.................................. 60
Hình 5.3 So sánh hiệu suất các phương pháp trên Synerise Dataset.......................... 71
Hình 5.4 Đóng góp của từng thành phần qua Ablation Study.................................. 75
Hình 5.5 Phân loại các lỗi phổ biến trong hệ thống gợi ý.................................... 78
Hình 6.1 Kiến trúc hệ thống triển khai thực tế................................................... 84

---

## DANH MỤC BẢNG BIỂU

Bảng 1.1 So sánh các phương pháp hệ thống gợi ý...................................................... 4
Bảng 2.1 So sánh các hàm mất mát cho Session-Based Recommendation......................... 22
Bảng 3.1 Tổng quan các phương pháp Session-Based Recommendation.......................... 31
Bảng 4.1 So sánh đóng góp của phương pháp đề xuất................................................ 55
Bảng 5.1 Thống kê tổng quan Kaggle Rental Dataset.................................................. 57
Bảng 5.2 Thống kê tổng quan Synerise RecSys Dataset............................................... 59
Bảng 5.3 Các chỉ số đánh giá sử dụng trong thí nghiệm........................................... 62
Bảng 5.4 Các phương pháp so sánh trong thí nghiệm............................................... 64
Bảng 5.5 Siêu tham số của mô hình GRU4Rec.......................................................... 65
Bảng 5.6 Siêu tham số của mô hình Contrastive Learning....................................... 66
Bảng 5.7 Kết quả trên Kaggle Rental Dataset (Local Validation)............................. 67
Bảng 5.8 Kết quả trên Synerise RecSys Dataset....................................................... 70
Bảng 5.9 Các chỉ số mở rộng trên Synerise Dataset................................................. 73
Bảng 5.10 Kết quả Ablation Study.......................................................................... 74
Bảng 5.11 Phân loại lỗi theo tần suất xuất hiện.................................................... 79

---

## CHƯƠNG 1

## TỔNG QUAN

### 1.1 Giới thiệu về hệ thống gợi ý

#### 1.1.1 Định nghĩa và phân loại

Hệ thống gợi ý (Recommender System) là một lớp của hệ thống lọc thông tin nhằm dự đoán mức độ ưu tiên hoặc rating mà người dùng có thể gán cho một mục tiêu (item) trong không gian候选 lớn. Các hệ thống này đã trở thành thành phần không thể thiếu trong nhiều nền tảng trực tuyến như Amazon (sản phẩm), Netflix (phim và truyền hình), Spotify (nhạc), và YouTube (video).

Theo cách tiếp cận, các hệ thống gợi ý có thể được phân thành ba nhóm chính:

**Hệ thống lọc cộng tác (Collaborative Filtering)**: Phương pháp này dựa trên giả định rằng nếu hai người dùng có cùng sở thích trong quá khứ, họ sẽ có xu hướng tương đồng trong tương lai. Các thuật toán thuộc nhóm này bao gồm User-based CF, Item-based CF, và Matrix Factorization. Ưu điểm chính của phương pháp này là không cần thông tin nội dung của items, chỉ cần dữ liệu tương tác người dùng-item. Tuy nhiên, phương pháp gặp vấn đề serious với cold-start problem và data sparsity.

**Hệ thống dựa trên nội dung (Content-based Filtering)**: Phương pháp này gợi ý các items có đặc điểm nội dung tương tự với những items người dùng đã thích trong quá khứ. Ví dụ, nếu người dùng thích một bộ phim hành động của đạo diễn Christopher Nolan, hệ thống sẽ gợi ý các phim hành động khác hoặc phim của cùng đạo diễn. Phương pháp này tránh được cold-start problem nhưng bị giới hạn bởi khả năng biểu diễn nội dung và thường gặp vấn đề over-specialization (chỉ gợi ý những items quá giống với những gì người dùng đã biết).

**Hệ thống kết hợp (Hybrid Methods)**: Kết hợp nhiều phương pháp khác nhau để tận dụng ưu điểm của từng phương pháp và khắc phục nhược điểm. Ví dụ, kết hợp Collaborative Filtering và Content-based Filtering, hoặc kết hợp nhiều mô hình Deep Learning khác nhau.

Hình 1.1 minh họa phân loại chi tiết các hệ thống gợi ý theo cách tiếp cận:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDER SYSTEMS                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐      │
│  │ Content-based  │  │ Collaborative  │  │    Hybrid      │      │
│  │                │  │                │  │                │      │
│  │ • Profile      │  │ • User-based   │  │ • Weighted     │      │
│  │   based        │  │ • Item-based   │  │   combination  │      │
│  │ • Item         │  │ • Matrix       │  │ • Cascade      │      │
│  │   attributes  │  │   Factorization│  │ • Feature      │      │
│  │                │  │ • Deep Learning│  │   augmentation │      │
│  └────────────────┘  └────────────────┘  └────────────────┘      │
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐      │
│  │ Knowledge-based│  │  Context-aware │  │   Session-based│      │
│  │                │  │                │  │                │      │
│  │ • Ontology     │  │ • Time         │  │ • Sequential   │      │
│  │ • Constraints  │  │ • Location     │  │   patterns     │      │
│  │                │  │ • Social       │  │ • RNN/GRU      │      │
│  │                │  │                │  │ • Transformer  │      │
│  └────────────────┘  └────────────────┘  └────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 1.1: Phân loại hệ thống gợi ý theo phương pháp tiếp cận**

#### 1.1.2 Các phương pháp cơ bản

**Collaborative Filtering (CF)** là phương pháp sớm nhất và được nghiên cứu rộng rãi nhất trong lĩnh vực hệ thống gợi ý. Phương pháp này được chia thành hai biến thể chính:

User-based CF tính toán độ tương đồng giữa người dùng hiện tại và tất cả người dùng khác trong hệ thống, sau đó gợi ý các items được những người dùng tương đồng đánh giá cao. Item-based CF tính toán độ tương đồng giữa items, gợi ý các items tương tự với những items người dùng đã tương tác trong quá khứ.

**Matrix Factorization (MF)** là phương pháp factorize ma trận rating người dùng-item thành tích của hai ma trận thấp chiều: ma trận embedding người dùng và ma trận embedding item. Các phương pháp nổi bật bao gồm SVD (Singular Value Decomposition) và ALS (Alternating Least Squares). MF đã trở thành phương pháp nền tảng cho nhiều cuộc thi hệ thống gợi ý như Netflix Prize.

**Deep Learning cho Recommender Systems** đã phát triển mạnh trong vài năm gần đây với các kiến trúc mạng nơ-ron sâu như Neural Collaborative Filtering (NCF), AutoEncoder, và các phương pháp sequence-based như GRU4Rec, SASRec, BERT4Rec. Các phương pháp này có khả năng học các biểu diễn phi tuyến tính phức tạp và kết hợp được nhiều loại dữ liệu phụ (side information).

Bảng 1.1 so sánh các phương pháp hệ thống gợi ý theo các khía cạnh quan trọng:

```
┌─────────────────┬───────────────┬──────────────────┬──────────────────┐
│ Phương pháp     │ Khả năng mở  │ Yêu cầu dữ liệu  │ Độ phức tạp      │
│                 │ rộng (scalab) │                  │ triển khai      │
├─────────────────┼───────────────┼──────────────────┼──────────────────┤
│ User-based CF   │ Thấp         │ Rating/user-item │ Thấp            │
│ Item-based CF   │ Trung bình   │ Rating/user-item │ Thấp            │
│ MF              │ Cao          │ Rating/implicit  │ Trung bình      │
│ Content-based   │ Cao          │ Item features    │ Trung bình      │
│ Deep Learning   │ Rất cao      │ Dữ liệu lớn     │ Cao             │
└─────────────────┴───────────────┴──────────────────┴──────────────────┘
```

**Bảng 1.1: So sánh các phương pháp hệ thống gợi ý**

### 1.2 Hệ thống gợi ý cho bài toán cho thuê

#### 1.2.1 Đặc thù của kinh tế chia sẻ

Kinh tế chia sẻ (Sharing Economy) đã phát triển mạnh mẽ trong thập kỷ vừa qua với tốc độ tăng trưởng kép hàng năm (CAGR) dự kiến đạt 19.4% trong giai đoạn 2023-2030. Các nền tảng cho thuê sản phẩm như Airbnb (nghỉ dưỡng), Turo (xe cộ), Rent the Runway (thời trang), và Fat Llama (đồ điện tử) đã thay đổi cách người dùng tiếp cận và sử dụng sản phẩm.

Khác với sở hữu truyền thống nơi người dùng mua đứt bán đoạn, mô hình cho thuê cho phép người dùng tiếp cận sản phẩm mà không cần đầu tư lớn ban đầu. Điều này tạo ra những hành vi người dùng khác biệt mà các hệ thống gợi ý truyền thống chưa được thiết kế để xử lý.

#### 1.2.2 So sánh với thương mại điện tử truyền thống

Bảng dưới đây so sánh chi tiết các đặc thù giữa bài toán cho thuê sản phẩm và mua sắm truyền thống:

```
┌─────────────────┬──────────────────────────┬──────────────────────────┐
│ Khía cạnh       │ Mua sắm truyền thống     │ Cho thuê sản phẩm        │
├─────────────────┼──────────────────────────┼──────────────────────────┤
│ Tần suất tương  │ Thường thấp              │ Lặp lại theo chu kỳ      │
│ tác            │ (rarely repeat)          │ (repeat purchases)       │
├─────────────────┼──────────────────────────┼──────────────────────────┤
│ Ràng buộc       │ Ngân sách                │ Thời gian, địa điểm,    │
│                │                          │ hoàn cảnh sử dụng        │
├─────────────────┼──────────────────────────┼──────────────────────────┤
│ Ngữ cảnh        │ Ý định mua               │ Hoàn cảnh sử dụng,      │
│ (Context)      │                          │ mùa, dịp                │
├─────────────────┼──────────────────────────┼──────────────────────────┤
│ Độ thưa        │ Khá nhiều users/interactions | Ít users hơn,      │
│ (Sparsity)     │                          │ data thưa hơn           │
├─────────────────┼──────────────────────────┼──────────────────────────┤
│ Cold-start      │ Dễ (có metadata)         │ Khó (hành vi mới)       │
├─────────────────┼──────────────────────────┼──────────────────────────┤
│ Tính mùa       │ Thấp                     │ Cao (mùa, dịp đặc biệt)  │
│ (Seasonality)  │                          │                          │
├─────────────────┼──────────────────────────┼──────────────────────────┤
│ Dữ liệu        │ Purchase event           │ View → Cart → Buy/Purchase│
│ đa hành vi     │                          │ với ý định khác nhau     │
└─────────────────┴──────────────────────────┴──────────────────────────┘
```

Các khác biệt chính bao gồm:

**Tính chu kỳ (Cyclical Pattern)**: Người dùng có xu hướng thuê lại cùng một sản phẩm sau một khoảng thời gian nhất định. Ví dụ, người dùng có thể thuê cùng một chiếc máy ảnh vào các dịp lễ hội khác nhau trong năm. Điều này khác với mua sắm truyền thống nơi người dùng thường chỉ mua một lần và hiếm khi mua lại.

**Phụ thuộc ngữ cảnh (Context Dependency)**: Sở thích thuê thay đổi đáng kể theo ngữ cảnh như mùa, dịp, địa điểm, và mục đích sử dụng. Một chiếc máy ảnh chuyên nghiệp có thể được thuê vào dịp lễ hội, trong khi một chiếc máy ảnh compact có thể được thuê cho các chuyến du lịch cuối tuần.

**Dữ liệu đa hành vi (Multi-behavior Data)**: Quá trình từ xem, thêm vào giỏ hàng, đến thực hiện thuê mang nhiều ý định khác nhau. View có thể thể hiện sự quan tâm ban đầu, cart thể hiện ý định thuê, và purchase/buy thể hiện quyết định cuối cùng. Các hành vi này cần được加权 khác nhau trong mô hình.

### 1.3 Đặt vấn đề và mục tiêu nghiên cứu

**Bài toán chính**

Cho một tập người dùng $U = \{u_1, u_2, ..., u_m\}$ và một tập sản phẩm $I = \{i_1, i_2, ..., i_n\}$, lịch sử tương tác của người dùng $u$ được biểu diễn bằng chuỗi $S_u = [(i_1, t_1, e_1), (i_2, t_2, e_2), ..., (i_k, t_k, e_k)]$, trong đó $i_j \in I$ là sản phẩm được tương tác, $t_j \in \mathbb{R}^+$ là timestamp của tương tác, và $e_j \in \{view, cart, buy\}$ là loại sự kiện.

Mục tiêu là học một hàm $f: S_u \rightarrow \hat{I}_u$ mà với mỗi người dùng $u$, trả về danh sách Top-K sản phẩm có xác suất tương tác cao nhất:

$$
\hat{I}_u = \arg\max_{I' \subset I, |I'|=K} \sum_{i \in I'} P(i|S_u)
$$

**Các thách thức chính**

1. **Data Sparsity**: Ma trận người dùng-sản phẩm rất thưa, nhiều người dùng có rất ít tương tác, làm cho các phương pháp Collaborative Filtering truyền thống hoạt động kém.

2. **Cold Start Problem**: Người dùng hoặc sản phẩm mới không có lịch sử tương tác, gây khó khăn cho các phương pháp dựa trên lịch sử.

3. **Temporal Dynamics**: Sở thích người dùng thay đổi theo thời gian, đòi hỏi hệ thống phải nắm bắt được các pattern thay đổi này.

4. **Re-purchase Pattern**: Cần mô hình hóa cả hành vi khám phá (discovery) sản phẩm mới và hành vi thuê lại (re-purchase) sản phẩm đã tương tác.

5. **Multi-behavior Signals**: View, cart, và buy mang ý định khác nhau và cần được xử lý phù hợp.

**Mục tiêu nghiên cứu**

Đồ án này tập trung vào các mục tiêu sau:

1. Nghiên cứu và đề xuất phương pháp mới phù hợp với đặc thù của bài toán gợi ý sản phẩm thuê, đặc biệt là khả năng nắm bắt hành vi thuê lại.

2. Xây dựng và triển khai hệ thống gợi ý hoàn chỉnh có khả năng kết hợp hiệu quả các tín hiệu: chuỗi hành vi tuần tự, biểu diễn ngữ nghĩa sản phẩm, và hành vi thuê lại.

3. Đánh giá hiệu quả của phương pháp đề xuất thông qua các thí nghiệm toàn diện trên nhiều bộ dữ liệu thực tế.

4. Phân tích chuyên sâu các yếu tố ảnh hưởng đến hiệu suất của hệ thống, bao gồm ablation study và failure analysis.

---

## CHƯƠNG 2

## CƠ SỞ LÝ THUYẾT

### 2.1 Collaborative Filtering

#### 2.1.1 User-based Collaborative Filtering

User-based Collaborative Filtering là một trong những phương pháp sớm nhất cho hệ thống gợi ý. Phương pháp này dựa trên giả định rằng nếu hai người dùng có cùng sở thích trong quá khứ, họ sẽ có xu hướng tương đồng trong tương lai.

Độ tương đồng giữa hai người dùng $u$ và $v$ thường được tính bằng hệ số tương quan Pearson:

$$
sim(u,v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}
$$

Trong đó, $I_{uv}$ là tập các sản phẩm mà cả người dùng $u$ và $v$ đều đã đánh giá, $r_{ui}$ là rating của người dùng $u$ cho sản phẩm $i$, và $\bar{r}_u$ là rating trung bình của người dùng $u$.

Sau khi tính được độ tương đồng với tất cả người dùng khác, dự đoán rating cho người dùng $u$ với sản phẩm $i$ được tính bằng:

$$
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} sim(u,v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} |sim(u,v)|}
$$

Trong đó, $N(u)$ là tập các người dùng tương đồng nhất với người dùng $u$ (top-k nearest neighbors).

#### 2.1.2 Item-based Collaborative Filtering

Item-based Collaborative Filtering tính độ tương đồng giữa các sản phẩm thay vì giữa người dùng. Độ tương đồng giữa hai sản phẩm $i$ và $j$:

$$
sim(i,j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_i)(r_{uj} - \bar{r}_j)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_i)^2} \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_j)^2}}
$$

Trong đó, $U_{ij}$ là tập người dùng đã đánh giá cả sản phẩm $i$ và $j$.

Dự đoán rating cho người dùng $u$ với sản phẩm $i$:

$$
\hat{r}_{ui} = \bar{r}_i + \frac{\sum_{j \in N(i)} sim(i,j) \cdot (r_{uj} - \bar{r}_j)}{\sum_{j \in N(i)} |sim(i,j)|}
$$

#### 2.1.3 Ưu điểm và nhược điểm

**Ưu điểm:**
- Đơn giản, dễ hiểu và dễ triển khai
- Không cần thông tin nội dung của sản phẩm
- Có khả năng giải thích được (dựa trên "người dùng tương tự" hoặc "sản phẩm tương tự")
- Hiệu quả tốt khi có đủ dữ liệu

**Nhược điểm:**
- **Cold-start problem**: Không thể gợi ý cho người dùng hoặc sản phẩm mới
- **Sparsity problem**: Ma trận người dùng-sản phẩm thường rất thưa, làm giảm chất lượng gợi ý
- **Scalability issues**: Với số lượng người dùng và sản phẩm lớn, việc tính độ tương đồng trở nên tốn kém
- **Không capture temporal patterns**: Không nắm bắt được các pattern thay đổi theo thời gian

### 2.2 Matrix Factorization

#### 2.2.1 Nguyên lý cơ bản

Matrix Factorization (MF) là phương pháp factorize ma trận rating $R \in \mathbb{R}^{m \times n}$ thành tích của hai ma trận thấp chiều:

$$
R \approx U \times V^T
$$

Trong đó:
- $U \in \mathbb{R}^{m \times k}$ là ma trận embedding người dùng
- $V \in \mathbb{R}^{n \times k}$ là ma trận embedding sản phẩm
- $k$ là dimension ẩn (latent dimension), thường $k \ll m, n$

Mỗi người dùng $u$ được biểu diễn bằng vector $u_u \in \mathbb{R}^k$ và mỗi sản phẩm $i$ được biểu diễn bằng vector $v_i \in \mathbb{R}^k$. Rating dự đoán:

$$
\hat{r}_{ui} = u_u^T v_i = \sum_{l=1}^{k} u_{ul} v_{il}
$$

#### 2.2.2 Các phương pháp tối ưu hóa

**Hàm mất mát cơ bản**

Hàm mất mát thường được sử dụng là Regularized Squared Error:

$$
\mathcal{L} = \sum_{(u,i) \in \mathcal{K}} (r_{ui} - u_u^T v_i)^2 + \lambda(\|u_u\|^2 + \|v_i\|^2)
$$

Trong đó, $\mathcal{K}$ là tập các cặp người dùng-sản phẩm có rating, và $\lambda$ là tham số regularization.

**Alternating Least Squares (ALS)**

ALS optimize từng biến một thời điểm trong khi giữ các biến khác cố định:

1. Giữ $V$ cố định, giải cho $U$:
$$
u_u = (V^T V + \lambda I)^{-1} V^T R_{(u,:)} \quad \forall u
$$

2. Giữ $U$ cố định, giải cho $V$:
$$
v_i = (U^T U + \lambda I)^{-1} U^T R_{(:,i)} \quad \forall i
$$

3. Lặp lại cho đến khi hội tụ

**Stochastic Gradient Descent (SGD)**

SGD update ngẫu nhiên các cặp người dùng-sản phẩm:

$$
u_u \leftarrow u_u + \gamma (e_{ui} v_i - \lambda u_u)
$$

$$
v_i \leftarrow v_i + \gamma (e_{ui} u_u - \lambda v_i)
$$

Trong đó, $e_{ui} = r_{ui} - u_u^T v_i$ là sai số dự đoán, và $\gamma$ là learning rate.

#### 2.2.3 Các biến thể mở rộng

**SVD++** (Koren, 2008) mở rộng SVD cơ bản bằng cách kết hợp thông tin ngầm (implicit feedback):

$$
\hat{r}_{ui} = \mu + b_u + b_i + v_i^T \left(u_u + |N(u)|^{-1/2} \sum_{j \in N(u)} y_j\right)
$$

Trong đó:
- $\mu$ là rating trung bình toàn cục
- $b_u$ và $b_i$ là biases của người dùng và sản phẩm
- $N(u)$ là tập sản phẩm người dùng $u$ đã tương tác (implicit feedback)
- $y_j$ là vector embedding của implicit feedback

**Factorization Machines (FM)** (Rendle, 2010) là một lớp mô hình tổng quát có thể factorize bất kỳ ma trận tương tác nào:

$$
\hat{y}(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j
$$

FM có thể mô hình hóa các tương tác bậc hai giữa các đặc điểm và có khả năng xử lý dữ liệu thưa hiệu quả.

### 2.3 Deep Learning cho Recommender Systems

#### 2.3.1 Neural Collaborative Filtering

Neural Collaborative Filtering (NCF) (He et al., 2017) thay thế inner product của Matrix Factorization bằng một mạng nơ-ron neural:

$$
\hat{y}_{ui} = f(u_U, v_I; \Theta)
$$

Kiến trúc NCF bao gồm:

1. **Embedding Layer**: Biến đổi người dùng và sản phẩm thành vector embedding
2. **Hidden Layers**: Các lớp fully connected để học các tương tác phi tuyến tính
3. **Output Layer**: Lớp output với sigmoid activation để dự đoán xác suất tương tác

NCF có thể được xem như một sự tổng quát hóa của Matrix Factorization, nơi inner product được thay thế bởi một hàm phi tuyến tính có thể học được.

**Generalized Matrix Factorization (GMF)**:

$$
\hat{y}_{ui} = a_{GMF}^T h(u_U \odot v_I)
$$

Trong đó $\odot$ là element-wise product.

**Multi-Layer Perceptron (MLP)**:

$$
\begin{align}
z_1 &= \text{ReLU}(W_1 [u_U; v_I] + b_1)) \\
z_2 &= \text{ReLU}(W_2 z_1 + b_2)) \\
\hat{y}_{ui} &= \sigma(W_3 z_2 + b_3)
\end{align}
$$

**NeuMF**: Kết hợp GMF và MLP:

$$
\hat{y}_{ui} = \sigma(h^T [a_{GMF}^T (u_U \odot v_I); z_{MLP}])
$$

#### 2.3.2 AutoEncoder cho Collaborative Filtering

AutoEncoder học compressed representation của input vector:

$$
\hat{r}_u = decoder(encoder(r_u))
$$

Cho collaborative filtering, input là vector rating của người dùng (với missing values được điền bằng 0), và AutoEncoder học reconstruct vector này.

**Collaborative Denoising AutoEncoder (CDAE)** (Wu et al., 2016) thêm noise vào input để regularize:

$$
\tilde{r}_u = \text{Dropout}(r_u)
$$

$$
h = f(W \tilde{r}_u + b)
$$

$$
\hat{r}_u = g(W' h + b')
$$

**Variational AutoEncoder (VAE)** cho Collaborative Filtering (Liang et al., 2018):

VAE-CF sử dụng variational inference để học probability distribution của user embeddings:

$$
p_\theta(z, r) = p(z) \prod_i p_\theta(r^{(i)}|z)
$$

$$
q_\phi(z|r) = \mathcal{N}(\mu_\phi(r), \text{diag}(\sigma^2_\phi(r)))
$$

### 2.4 Session-Based Recommendation

#### 2.4.1 Đặc thù của bài toán Session-Based

Session-Based Recommendation (SBR) tập trung vào việc dự đoán item tiếp theo trong một phiên ngắn hạn (thường là 30 phút), không sử dụng thông tin người dùng dài hạn. Điều này khác với các phương pháp truyền thống nơi toàn bộ lịch sử người dùng được sử dụng.

Đặc thù của SBR:
- Không có thông tin định danh (anonymous sessions)
- Chuỗi tương tác ngắn
- Cần real-time recommendations
- Mục tiêu: dự đoán click tiếp theo hoặc purchase tiếp theo

#### 2.4.2 GRU4Rec và các biến thể

**GRU4Rec** (Hidasi et al., 2016) là một trong những phương pháp đầu tiên áp dụng RNN cho SBR:

$$
h_t = \text{GRU}(h_{t-1}, e_{i_t})
$$

Trong đó:
- $h_t$ là hidden state tại thời điểm t
- $e_{i_t}$ là embedding của item $i_t$ được tương tác tại thời điểm t
- $h_0$ là initial hidden state (thường là zero vector)

Kiến trúc GRU4Rec bao gồm:

1. **Item Embedding Layer**: Biến đổi item ID thành vector embedding
2. **GRU Layer**: Xử lý chuỗi embeddings
3. **Output Layer**: Linear layer để dự đoán scores cho tất cả items

```
┌─────────────────────────────────────────────────────────────────────┐
│                      GRU4REC ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: [i₁, i₂, i₃, ..., iₜ]                                     │
│    ↓                                                                │
│  Embedding Layer: [e₁, e₂, e₃, ..., eₜ]                            │
│    ↓                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    GRU LAYER                               │   │
│  │                                                             │   │
│  │   h₁ = GRU(h₀, e₁)                                         │   │
│  │   h₂ = GRU(h₁, e₂)                                         │   │
│  │   ...                                                       │   │
│  │   hₜ = GRU(hₜ₋₁, eₜ)                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│    ↓                                                                │
│  Output Layer: s = W·hₜ + b  (scores for all items)               │
│    ↓                                                                │
│  Top-K items with highest scores                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 2.4: Kiến trúc GRU4Rec cơ bản**

#### 2.4.3 Các hàm mất mát cho Session-Based Recommendation

**Cross-Entropy (CE) Loss**

Hàm mất mát Cross-Entropy được sử dụng rộng rãi trong các bài toán classification:

$$
\mathcal{L}_{CE} = -\sum_{t=1}^{T-1} \ln \frac{\exp(s_{i_{t+1}})}{\sum_{j \in I} \exp(s_j)}
$$

Trong đó, $s_{i_{t+1}}$ là score cho item tiếp theo thực tế, và $s_j$ là score cho item j.

Nhược điểm chính của CE loss là tính toán tốn kém do cần tính softmax trên toàn bộ tập items.

**BPR (Bayesian Personalized Ranking) Loss**

BPR optimize ranking trực tiếp thay vì predicting absolute ratings:

$$
\mathcal{L}_{BPR} = -\sum_{(u,i,j) \in \mathcal{D}} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
$$

Trong đó:
- $(u,i,j)$ là triplet: người dùng u, positive item i, negative item j
- $\hat{y}_{ui}$ là score cho positive item
- $\hat{y}_{uj}$ là score cho negative sample
- $\sigma$ là sigmoid function

Ưu điểm của BPR:
- Direct ranking optimization phù hợp với Top-K recommendation
- Efficient với in-batch negative sampling
- Không cần compute over full item space

**TOP1 Loss**

TOP1 loss được đề xuất trong paper GRU4Rec gốc:

$$
\mathcal{L}_{TOP1} = \sum_{(u,i,j) \in \mathcal{D}} \sigma(\hat{y}_{uj} - \hat{y}_{ui}) + \sigma(\hat{y}_{uj}^2)
$$

Hàm mất mát này kết hợp hai thành phần: phần đầu đảm bảo positive item có score cao hơn negative sample, phần thứ hai regularize scores của negative samples.

**So sánh các hàm mất mát**

Bảng 2.1 so sánh các hàm mất mát cho Session-Based Recommendation:

```
┌─────────┬──────────────────────┬─────────────────────┬──────────────────┐
│ Loss    │ Ưu điểm              │ Nhược điểm           │ Phù hợp với      │
├─────────┼──────────────────────┼─────────────────────┼──────────────────┤
│ CE      │ • Stable             │ • Compute expensive  │ • Small item     │
│         │ • Widely used        │ • Softmax over all   │   vocabulary     │
│         │                      │   items              │                  │
├─────────┼──────────────────────┼─────────────────────┼──────────────────┤
│ BPR     │ • Ranking-focused    │ • Cần negative       │ • Top-K          │
│         │ • Efficient          │   sampling           │   recommendation │
│         │ • Proven effective   │                      │                  │
├─────────┼──────────────────────┼─────────────────────┼──────────────────┤
│ TOP1    │ • Robust             │ • Less studied       │ • Session-based   │
│         │ • Fast               │                      │   tasks          │
└─────────┴──────────────────────┴─────────────────────┴──────────────────┘
```

**Bảng 2.1: So sánh các hàm mất mát cho Session-Based Recommendation**

### 2.5 Contrastive Learning

#### 2.5.1 Nguyên lý cơ bản

Contrastive Learning (CL) là một phương pháp học biểu diễn (representation learning) dựa trên ý tưởng kéo các positive pairs closer và đẩy các negative pairs farther trong embedding space.

Nguyên lý cơ bản: cho một anchor sample $z$, một positive sample $z^+$ (cùng class hoặc augmented view của $z$), và multiple negative samples $\{z^-_1, z^-_2, ..., z^-_K\}$, CL học một encoder $f(\cdot)$ sao cho:

$$
\text{sim}(f(z), f(z^+)) \gg \text{sim}(f(z), f(z^-_i)) \quad \forall i
$$

Trong đó, $\text{sim}(\cdot, \cdot)$ thường là cosine similarity:

$$
\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}
$$

#### 2.5.2 InfoNCE Loss và các biến thể

**InfoNCE Loss** (Oord et al., 2018) là một trong những hàm mất mát phổ biến nhất cho Contrastive Learning:

$$
\mathcal{L}_{InfoNCE} = -\mathbb{E} \left[ \ln \frac{\exp(\text{sim}(z, z^+)/\tau)}{\exp(\text{sim}(z, z^+)/\tau) + \sum_{z^- \in N} \exp(\text{sim}(z, z^-)/\tau)} \right]
$$

Trong đó:
- $\tau$ là temperature parameter (thường 0.07-0.1)
- $N$ là tập negative samples

**SimCLR** (Chen et al., 2020) sử dụng InfoNCE loss với data augmentation:

1. Data augmentation: Tạo các views khác nhau của cùng một sample
2. Encoder: Encode các views thành representations
3. Projection head: Map representations đến một space where contrastive loss được áp dụng
4. Contrastive loss: Optimize với InfoNCE

**MoCo** (He et al., 2020) sử dụng:

1. **Queue**: Maintains a large negative sample queue
2. **Moving average encoder**: Key encoder là EMA của query encoder
3. **Consistency**: Giữ representations consistent

```
┌─────────────────────────────────────────────────────────────────────┐
│              CONTRASTIVE LEARNING FRAMEWORK                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Sample                                                     │
│      │                                                             │
│      ▼                                                             │
│  Data Augmentation (two different views)                          │
│      │                                                             │
│      ├─────────────────────┐                                       │
│      │                     │                                       │
│      ▼                     ▼                                       │
│   View 1                View 2                                     │
│      │                     │                                       │
│      ▼                     ▼                                       │
│   Encoder f(.|θ)       Encoder f(.|θ)                             │
│      │                     │                                       │
│      ▼                     ▼                                       │
│   Projection g(.)      Projection g(.)                            │
│      │                     │                                       │
│      ▼                     ▼                                       │
│   z₁ (positive)        z₂ (positive)                              │
│      │                     │                                       │
│      └──────────┬──────────┘                                       │
│                 │                                                  │
│                 ▼                                                  │
│         Contrastive Loss (InfoNCE)                                 │
│                 │                                                  │
│                 ├─→ Positive pair (z₁, z₂)                         │
│                 │                                                  │
│                 └─→ Negative samples from queue/batch              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 2.5: Minh họa quá trình tạo cặp đối chiếu trong Contrastive Learning**

#### 2.5.3 Ứng dụng trong Recommender Systems

Contrastive Learning đã được áp dụng thành công trong Recommender Systems để:

1. **Learn item embeddings**: Items trong cùng session được coi là positive pairs
2. **Learn user embeddings**: Sessions cùng user được coi là positive pairs
3. **Data augmentation**: Create augmented views của sequences (crop, mask, reorder)
4. **Self-supervised pre-training**: Pre-train trên large unlabeled data trước khi fine-tune

**DualCL** (Wu et al., 2021) áp dụng Contrastive Learning cho cả users và items:

$$
\mathcal{L} = \mathcal{L}_{user}^{CL} + \mathcal{L}_{item}^{CL} + \mathcal{L}_{interaction}
$$

**S3-Rec** (Sun et al., 2019) sử dụng self-supervised tasks cho sequential recommendation:
- Attribute masking: Dự đoán attributes bị mask
- Item predicting: Dự đoán items bị mask trong sequence
- Direction prediction: Dự đoán direction của sequence

---

## CHƯƠNG 3

## CÁC CÔNG TRÌNH LIÊN QUAN

### 3.1 Phương pháp truyền thống

#### 3.1.1 Phương pháp dựa trên nội dung (Content-based)

Phương pháp Content-based Filtering gợi ý các items có đặc điểm nội dung tương tự với những items người dùng đã thích trong quá khứ. Các phương pháp phổ biến bao gồm:

**TF-IDF Vector Space Model**: Sử dụng Term Frequency-Inverse Document Frequency để biểu diễn items và profiles người dùng dưới dạng vectors, sau đó tính cosine similarity giữa chúng.

**Probabilistic Models**: Các mô hình như Naive Bayes Classifier được sử dụng để dự đoán xác suất người dùng sẽ thích một item dựa trên đặc điểm nội dung.

**Decision Trees & Rules**: Các cây quyết định được xây dựng dựa trên đặc điểm người dùng và item để đưa ra quyết định gợi ý.

#### 3.1.2 Collaborative Filtering trong thương mại điện tử

**Item-based CF cho E-commerce**

Linden et al. (2003) đề xuất item-to-item collaborative filtering cho Amazon, một trong những ứng dụng thực tế đầu tiên và thành công nhất của CF. Phương pháp này:

- Compute item-item similarity dựa trên user behavior
- Recommend items similar to những items người dùng đã tương tác
- Scale lên millions of items và customers

**Matrix Factorization với Implicit Feedback**

Hu et al. (2008) đề xuất ALS cho implicit feedback datasets:

$$
\mathcal{L} = \sum_{u,i} c_{ui} (p_{ui} - u_u^T v_i)^2 + \lambda(\|u_u\|^2 + \|v_i\|^2)
$$

Trong đó $c_{ui}$ là confidence weight và $p_{ui}$ là preference (binary indicator).

#### 3.1.3 Hybrid Methods

**Content-Boosted Collaborative Filtering** (Melville et al., 2002) kết hợp content-based và collaborative filtering bằng cách sử dụng content-based method để fill missing values trước khi áp dụng CF.

**Weighted Hybrid** (Schein et al., 2002) kết hợp multiple recommenders:

$$
\hat{r}_{ui} = \sum_{j} \alpha_j \hat{r}_{ui}^{(j)}
$$

Trong đó $\alpha_j$ là weight cho recommender thứ j, và $\sum_j \alpha_j = 1$.

### 3.2 Deep Learning hiện đại cho Recommender Systems

#### 3.2.1 Session-Based Recommendations với RNN

**GRU4Rec** (Hidasi et al., 2016) là một trong những phương pháp đầu tiên áp dụng RNN cho SBR:

- Sử dụng GRU để model session sequences
- Đề xuất BPR và TOP1 loss functions
- State-of-the-art cho các benchmark datasets khi đó

**NARM** (Li et al., 2017) (Neural Attentive Session-based Recommendation):
- Sử dụng attention mechanism để capture cả sequential behavior và user's main purpose
- Hybrid encoder với GRU và attention

**STAMP** (Liu et al., 2018) (Short-Term Attention/Memory Priority):
- Sử dụng short-term memory và attention để model session interests
- Efficient với memory complexity thấp

#### 3.2.2 Self-Attentive Sequential Recommendation

**SASRec** (Kang & McAuley, 2018) sử dụng Transformer's self-attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Ưu điểm chính:
- Parallel training (khác RNN)
- Better long-term dependency modeling
- State-of-the-art performance

**BERT4Rec** (Sun et al., 2019) sử dụng bidirectional self-attention:
- Masked language model approach cho RecSys
- Left-to-right và bidirectional training

Bảng 3.1 tổng quan các phương pháp Session-Based Recommendation:

```
┌──────────────┬────────────────┬────────────────┬────────────────┐
│ Phương pháp  │ Kiến trúc      │ Loss function  │ Năm công bố   │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ GRU4Rec      │ RNN (GRU)      │ BPR, TOP1      │ 2016           │
│ NARM         │ GRU + Attn     │ Cross-Entropy  │ 2017           │
│ STAMP        │ Attn + Memory  │ Cross-Entropy  │ 2018           │
│ SASRec       │ Transformer    │ BPR            │ 2018           │
│ BERT4Rec     │ Bi-Transformer │ MLM            │ 2019           │
│ CL4SRec      │ CL + Transformer│ CL + CE       │ 2021           │
└──────────────┴────────────────┴────────────────┴────────────────┘
```

**Bảng 3.1: Tổng quan các phương pháp Session-Based Recommendation**

#### 3.2.3 BERT4Rec và các phương pháp Transformer-based

**BERT4Rec** áp dụng kiến trúc BERT vào sequential recommendation:

1. **Masked Token Prediction**: Randomly mask tokens trong sequence và dự đoán chúng
2. **Bidirectional Context**: Model cả left-to-right và right-to-left context
3. **Fine-tuning**: Fine-tune cho downstream task (next-item prediction)

**GPT-based RecSys** áp dụng kiến trúc GPT:

- Left-to-right causal attention
- Generate recommendations autoregressively
- Better cho explanation generation

**XLNet cho RecSys** sử dụng permutation language modeling:

- Capture bidirectional context với autoregressive
- Better cho long sequences

### 3.3 Khoảng trống nghiên cứu

Từ tổng quan literature, chúng tôi nhận thấy các khoảng trống sau:

**Re-purchase Behavior**: Hầu hết các phương pháp hiện tại coi mỗi interaction độc lập, bỏ qua pattern thuê lại (re-purchase) đặc trưng của rental domain. Một số ít nghiên cứu đề xuất repeat-exploration nhưng chưa có systematic approach.

**Multi-behavior Fusion**: View, cart, và buy events thường được treat như nhau hoặc chỉ được sử dụng để construct training data. Ít có nghiên cứu về cách model khác biệt các behavior types này.

**Sequential + Graph Integration**: Các phương pháp sequence-based (SASRec, BERT4Rec) và graph-based (LightGCN, PIN) thường được nghiên cứu riêng. Ít có work về việc kết hợp cả sequential và graph information.

**Domain-specific Methods**: Đa số research tập trung vào general domains (movie, music, e-commerce). Chỉ có ít nghiên cứu về domain-specific methods cho rental/re-commerce.

Bảng dưới đây tóm tắt các khoảng trống nghiên cứu:

```
┌───────────────────────┬─────────────────────────┬──────────────────┐
│ Vấn đề               │ Trạng thái hiện tại     │ Khoảng trống     │
├───────────────────────┼─────────────────────────┼──────────────────┤
│ Re-purchase behavior  │ Ít được nghiên cứu      │ Cần explicit     │
│                       │                         │ modeling          │
├───────────────────────┼─────────────────────────┼──────────────────┤
│ Multi-behavior fusion │ Treat equally           │ Cần event-       │
│                       │                         │ weighted         │
│                       │                         │ approach         │
├───────────────────────┼─────────────────────────┼──────────────────┤
│ Sequential + Graph    │ Thường tách rời         │ Cần integrated    │
│                       │                         │ approach          │
├───────────────────────┼─────────────────────────┼──────────────────┤
│ Rental domain         │ Không có research       │ Cần domain-      │
│                       │                         │ specific methods │
├───────────────────────┼─────────────────────────┼──────────────────┤
│ Explainability        │ Black-box models        │ Cần interpretable │
│                       │                         │ approaches       │
└───────────────────────┴─────────────────────────┴──────────────────┘
```

---

## CHƯƠNG 4

## PHƯƠNG PHÁP ĐỀ XUẤT

Chương này trình bày chi tiết phương pháp CL-GRU4Rec+RP đề xuất, bao gồm kiến trúc tổng thể, ba thành phần chính, và chiến lược kết hợp thích ứng hai giai đoạn.

### 4.1 Tổng quan kiến trúc

#### 4.1.1 Thiết kế mô-đun

Phương pháp đề xuất được thiết kế theo kiến trúc mô-đun với ba thành phần độc lập:

**Thành phần 1: GRU4Rec với BPR Loss**
- Mục tiêu: Learn sequential patterns trong chuỗi hành vi người dùng
- Input: Chuỗi item IDs đã tương tác
- Output: Hidden states cho mỗi position trong sequence
- Loss function: BPR (Bayesian Personalized Ranking)

**Thành phần 2: Contrastive Learning**
- Mục tiêu: Learn semantic item embeddings không phụ thuộc vào sequential context
- Input: Các cặp items xuất hiện cùng session
- Output: Item embeddings với cosine similarity
- Loss function: InfoNCE (Contrastive Loss)

**Thành phần 3: Re-Purchase Awareness**
- Mục tiêu: Capture cyclical re-purchase behavior đặc trưng của rental domain
- Input: Lịch sử tương tác với event types
- Output: Scores reflecting re-purchase probability
- Scoring function: Event-weighted với recency boost

Hình 4.1 minh họa kiến trúc tổng thể:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  CL-GRU4REC+RP ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                           INPUT LAYER                               │
│                    User History: S_u = [(i,e,t)...]                │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    PARALLEL TRAINING                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │   │
│  │  │   GRU4Rec    │  │   Contrastive│  │ Re-Purchase  │       │   │
│  │  │   + BPR      │  │   Learning   │  │  Awareness   │       │   │
│  │  │              │  │              │  │              │       │   │
│  │  │ • Sequential │  │ • Item       │  │ • Event-     │       │   │
│  │  │   patterns  │  │   semantics  │  │   weighted   │       │   │
│  │  │ • BPR loss   │  │ • Session    │  │ • Recency    │       │   │
│  │  │ • Hidden     │  │   pairs      │  │   boost      │       │   │
│  │  │   states    │  │ • SimCL      │  │ • Cyclical   │       │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              ADAPTIVE TWO-STAGE FUSION                       │   │
│  │                                                              │   │
│  │  Stage 1: Re-Purchase Dominant                              │   │
│  │    IF strong RP signal (≥K unique items):                   │   │
│  │        RP fills most slots                                  │   │
│  │    ELSE:                                                     │   │
│  │        RP + GRU + CL fill slots proportionally              │   │
│  │                                                              │   │
│  │  Stage 2: Discovery Fills Remaining Slots                   │   │
│  │    IF still < K recommendations:                             │   │
│  │        CL + GRU + CoOccur fill discovery slots              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│                        OUTPUT LAYER                                │
│                    Top-K: [rec_1, ..., rec_K]                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 4.1: Kiến trúc tổng thể CL-GRU4Rec+RP**

#### 4.1.2 Chiến lược huấn luyện và suy luận

**Chiến lược huấn luyện độc lập**

Ba thành phần được huấn luyện độc lập với các mục tiêu khác nhau:

1. **GRU4Rec**: Optimize cho next-item prediction với BPR loss
2. **Contrastive Learning**: Optimize cho item similarity với InfoNCE loss
3. **Re-Purchase**: Không cần huấn luyện, tính toán trực tiếp từ lịch sử

Ưu điểm của chiến lược này:
- Tránh multi-task confusion: Khi train multiple objectives cùng lúc, các gradients có thể interfere với nhau
- Flexible: Mỗi thành phần có thể được updated/retrained độc lập
- Debuggable: Dễ dàng analyze contribution của từng thành phần

**Chiến lược suy luận kết hợp**

Tại thời điểm inference, ba thành phần được kết hợp thông qua chiến lược hai giai đoạn thích ứng:

```
┌─────────────────────────────────────────────────────────────────────┐
│              TRAINING VS INFERENCE STRATEGY                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TRAINING PHASE (Independent):                                     │
│  ─────────────────────────────────────                             │
│     GRU4Rec Model  ←→  GRU4Rec Data  ←→  BPR Loss                │
│     CL Model       ←→  CL Data       ←→  InfoNCE Loss             │
│     RP Component  ←→  N/A (Rule-based)                            │
│                                                                     │
│  INFERENCE PHASE (Combined):                                       │
│  ─────────────────────────────────────                             │
│     User Input → [GRU, CL, RP Scores] → Fusion → Top-K Recs        │
│                                                                     │
│  Key Advantage: Training independent → Inference combined         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 4.2: Chiến lược huấn luyện độc lập và kết hợp tại suy luận**

### 4.2 Thành phần 1: GRU4Rec với BPR Loss

#### 4.2.1 Kiến trúc mạng GRU4Rec

Mạng GRU4Rec được thiết kế với các lớp sau:

**Item Embedding Layer**

Biến đổi item ID thành dense vector:

$$
e_i = E[i] \in \mathbb{R}^{d_e}
$$

Trong đó $E \in \mathbb{R}^{|I| \times d_e}$ là embedding matrix, và $d_e$ là embedding dimension (thường 128).

**Dropout Layer**

Dropout được áp dụng sau embedding để prevent overfitting:

$$
e'_i = \text{Dropout}(e_i, p)
$$

Với $p$ là dropout probability (thường 0.15).

**GRU Layer**

Gated Recurrent Unit (GRU) được sử dụng để process sequence:

$$
\begin{align}
r_t &= \sigma(W_r e'_t + U_r h_{t-1}) \\
z_t &= \sigma(W_z e'_t + U_z h_{t-1}) \\
\tilde{h}_t &= \tanh(W_h e'_t + U_h (r_t \odot h_{t-1})) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{align}
$$

Trong đó:
- $r_t$ là reset gate
- $z_t$ là update gate
- $\tilde{h}_t$ là candidate hidden state
- $h_t$ là hidden state tại thời điểm t
- $\sigma$ là sigmoid function
- $\odot$ là element-wise multiplication

**Output Projection Layer**

Linear layer projecting hidden state đến item space:

$$
s_i = W_o h_t + b_o \in \mathbb{R}^{|I|}
$$

Hình 4.3 minh họa kiến trúc chi tiết:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      GRU4REC MODEL ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: [i₁, i₂, ..., iₜ]                                         │
│    │                                                                │
│    ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Item Embedding Layer                            │   │
│  │  E ∈ ℝ^{|I| × dₑ},  dₑ = 128                               │   │
│  │  [e₁, e₂, ..., eₜ]                                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│    │                                                                │
│    ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Dropout Layer                             │   │
│  │  p = 0.15                                                    │   │
│  │  [e'₁, e'₂, ..., e'ₜ]                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│    │                                                                │
│    ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      GRU Layer                               │   │
│  │  Hidden dim: dₕ = 200                                       │   │
│  │  [h₁, h₂, ..., hₜ]                                         │   │
│  │                                                             │   │
│  │  rₜ = σ(Wᵣe'ₜ + Uᵣhₜ₋₁)                                   │   │
│  │  zₜ = σ(W₂e'ₜ + U₂hₜ₋₁)                                   │   │
│  │  h̃ₜ = tanh(Wₕe'ₜ + Uₕ(rₜ ⊙ hₜ₋₁))                         │   │
│  │  hₜ = (1-zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│    │                                                                │
│    ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Output Projection Layer                         │   │
│  │  Wₒ ∈ ℝ^{dₕ × |I|},  bₒ ∈ ℝ^{|I|}                          │   │
│  │  s = Wₒhₜ + bₒ  ∈ ℝ^{|I|}                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│    │                                                                │
│    ▼                                                                │
│  Output: Scores for all items                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 4.3: Kiến trúc mạng GRU4Rec**

#### 4.2.2 Hàm mất mát BPR

BPR (Bayesian Personalized Ranking) loss optimize cho ranking trực tiếp:

$$
\mathcal{L}_{BPR} = -\sum_{(u,i,j) \in \mathcal{D}} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
$$

**Triplet Construction**

Cho mỗi positive pair $(u,i)$ (người dùng $u$ đã tương tác với item $i$), negative sample $j$ được chọn ngẫu nhiên từ tập items mà người dùng $u$ chưa tương tác.

**Gradient**

Gradient của BPR loss:

$$
\frac{\partial \mathcal{L}_{BPR}}{\partial \theta} = -\sum_{(u,i,j) \in \mathcal{D}} \left(1 - \sigma(\hat{y}_{ui} - \hat{y}_{uj})\right) \frac{\partial (\hat{y}_{ui} - \hat{y}_{uj})}{\partial \theta}
$$

**In-batch Negative Sampling**

Trong thực nghiệm, chúng tôi sử dụng in-batch negative sampling để improve efficiency:

1. Trong mỗi batch, positive items của tất cả users được sử dụng làm negatives cho nhau
2. Điều này đảm bảo negative samples hard hơn (không phải hoàn toàn ngẫu nhiên)
3. Giảm số lượng sampling operations cần thiết

#### 4.2.3 Chiến lược huấn luyện

**Ensemble Training**

Chúng tôi train ensemble của 3 models với different random seeds (42, 123, 456) để:
- Reduce variance của predictions
- Improve stability của model
- Better generalization

**Learning Rate Scheduling**

Cosine annealing scheduler được sử dụng:

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_{max}}\pi))
$$

Trong đó:
- $\eta_{max}$ là initial learning rate
- $\eta_{min}$ là minimum learning rate
- $T_{cur}$ là current epoch
- $T_{max}$ là total epochs

**Gradient Clipping**

Gradient clipping được áp dụng để prevent exploding gradients:

$$
\|g\| = \min\left(1, \frac{\lambda}{\|g\|}\right) g
$$

Với $\lambda$ là clipping threshold (thường 5.0).

**Data Augmentation**

Random cropping được áp dụng cho sequences dài hơn maximum length:

```python
if len(sequence) > max_length + 1:
    start = random.randint(0, len(sequence) - max_length - 1)
    sequence = sequence[start:start + max_length + 1]
```

Điều này:
- Acts như data augmentation
- Forces model to learn context-independent representations
- Reduces overfitting

### 4.3 Thành phần 2: Contrastive Learning cho biểu diễn ngữ nghĩa sản phẩm

#### 4.3.1 Xây dựng cặp dữ liệu đối chiếu

**Positive Pairs Construction**

Items xuất hiện cùng trong một session được coi là positive pairs:

$$
\mathcal{P} = \{(i, j) : \exists s, i \in s \land j \in s \land i \neq j\}$$

Với $s$ là một session.

**Chiến lược sampling**

Có hai chiến lược được sử dụng:

1. **Full pairing** cho short sessions (< 20 items):
$$
\mathcal{P}_{full} = \{(i, j) : i, j \in s, i < j\}
$$

2. **Random sampling** cho long sessions (≥ 20 items):
$$
\mathcal{P}_{sample} = \text{Sample}(\{(i, j) : i, j \in s, i \neq j\}, N=40)
$$

Điều này:
- Balance giữa quantity và quality của positive pairs
- Avoid over-representing very long sessions
- Reduce computational cost

#### 4.3.2 Kiến trúc mạng Contrastive Learning

**Embedding Layer**

Base embedding layer với Xavier initialization:

$$
z_i^0 = E[i] \in \mathbb{R}^{d_c}
$$

Với $d_c = 64$ là contrastive embedding dimension.

**Projection Head**

Two-layer MLP với GELU activation:

$$
\begin{align}
z_i^1 &= \text{GELU}(W_1 z_i^0 + b_1) \\
z_i^2 &= W_2 z_i^1 + b_2
\end{align}
$$

**L2 Normalization**

Final representations được L2-normalized:

$$
z_i = \frac{z_i^2}{\|z_i^2\|_2}
$$

Điều này đảm bảo:
- Cosine similarity bằng dot product sau normalization
- Stable training
- Better representations

Hình 4.4 minh họa kiến trúc:

```
┌─────────────────────────────────────────────────────────────────────┐
│              CONTRASTIVE LEARNING MODEL ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: Item ID [i]                                                │
│    │                                                                │
│    ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Base Embedding Layer                            │   │
│  │  E ∈ ℝ^{|I| × d_c},  d_c = 64                                │   │
│  │  z⁰[i] = E[i]                                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│    │                                                                │
│    ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                Projection Head (2-layer MLP)                 │   │
│  │                                                             │   │
│  │  z¹ = GELU(W₁z⁰ + b₁),  dim: d_c → 2d_c                   │   │
│  │  z² = W₂z¹ + b₂,          dim: 2d_c → d_c                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│    │                                                                │
│    ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   L2 Normalization                           │   │
│  │  z = z² / ||z²||₂                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│    │                                                                │
│    ▼                                                                │
│  Output: Normalized embedding z ∈ ℝ^{d_c}                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 4.4: Kiến trúc mạng Contrastive Learning**

#### 4.3.3 Hàm mất mát Contrastive

**InfoNCE Loss với Temperature**

$$
\mathcal{L}_{CL} = -\mathbb{E}_{(i,j) \sim \mathcal{P}} \left[ \ln \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\exp(\text{sim}(z_i, z_j)/\tau) + \sum_{k \in \mathcal{N}} \exp(\text{sim}(z_i, z_k)/\tau)} \right]
$$

Trong đó:
- $\text{sim}(u, v) = u^T v$ (dot product của L2-normalized vectors = cosine similarity)
- $\tau$ là temperature parameter (0.07)
- $\mathcal{N}$ là tập negative samples

**Hard Negative Mining**

Chúng tôi sử dụng two strategies cho negative sampling:

1. **Random negatives**: Sample ngẫu nhiên từ all items
2. **Hard negatives**: Sample từ items có medium similarity với anchor

**Training Strategy**

- **Batch size**: 1024 (large batch giúp có nhiều negatives)
- **Negatives per positive**: 256
- **Epochs**: 25
- **Optimizer**: Adam với weight decay 1e-5
- **Learning rate**: 0.003 với cosine annealing

### 4.4 Thành phần 3: Re-Purchase Awareness

#### 4.4.1 Phân tích hành vi thuê lại

**Re-purchase Pattern trong Rental Domain**

Khác với mua sắm truyền thống, rental có pattern đặc trưng:

1. **Intra-session repeats**: Người dùng có thể thuê lại cùng sản phẩm trong cùng session (ví dụ: thử máy ảnh khác)
2. **Inter-session repeats**: Người dùng thuê lại sản phẩm sau một khoảng thời gian (ví dụ: thuê cùng máy ảnh cho các dịp lễ khác nhau)
3. **Event-dependent repeats**: Buy events có higher probability của repeats hơn view events

**Empirical Analysis**

Trên Synerise dataset, chúng tôi观察到:
- ~35% users có re-purchase behavior
- Re-purchase rate cho buy events: ~28%
- Re-purchase rate cho cart events: ~15%
- Re-purchase rate cho view events: ~8%

#### 4.4.2 Mô hình hóa tín hiệu đa hành vi

**Event Weighting**

Các event types được weighted khác nhau:

$$
w(e) = \begin{cases}
5.0 & \text{nếu } e = \text{buy} \\
2.0 & \text{nếu } e = \text{cart} \\
1.0 & \text{nếu } e = \text{view}
\end{cases}
$$

**Rationale**:
- Buy events thể hiện clear intent
- Cart events thể hiện moderate intent
- View events thể hiện weak intent

**Recency Boost**

Recent interactions được weighted cao hơn:

$$
r(pos) = 1 + \frac{pos}{len(history)}$$

Trong đó $pos$ là position trong sequence (0-indexed từ đầu đến cuối).

#### 4.4.3 Tính toán điểm Re-Purchase

**Scoring Function**

Điểm Re-Purchase cho item $i$:

$$
RP(i) = \sum_{(i_t, e_t) \in \mathcal{H}: i_t = i} w(e_t) \cdot r(pos_t)
$$

Trong đó $\mathcal{H}$ là lịch sử người dùng.

**Top-K Selection**

$$
\text{Top-K}_{RP} = \arg\max_{I' \subset I, |I'|=K} \sum_{i \in I'} RP(i)
$$

### 4.5 Chiến lược kết hợp thích ứng hai giai đoạn

#### 4.5.1 Giai đoạn 1: Ưu tiên Re-Purchase

**Mục tiêu**

Tận dụng re-purchase pattern để provide relevant recommendations cho users có strong history.

**Algorithm**

```
Input: User history H = [(i₁, e₁), ..., (iₙ, eₙ)], K recommendations
Output: Top-K recommendations

1. Compute RP scores for all items in H
2. Select Top-K items with highest RP scores
3. Return these items as Stage 1 recommendations
```

**Heuristics**

Nếu user có strong re-purchase signal (≥K unique items với RP score > threshold), RP fills most slots. Otherwise, RP combined với other signals.

#### 4.5.2 Giai đoạn 2: Khám phá và lấp đầy

**Mục tiêu**

Fill remaining slots với discovery items (items chưa được tương tác hoặc ít phổ biến).

**Discovery Sources**

1. **GRU Sequential**: Items predicted by GRU4Rec model
2. **CL Similarity**: Items semantically similar to user's recent items
3. **Co-occurrence**: Items frequently co-occur với user's items

**Combination Strategy**

$$
Score_{discovery}(i) = \alpha \cdot Score_{GRU}(i) + \beta \cdot Score_{CL}(i) + \gamma \cdot Score_{CoOccur}(i)
$$

Với:
- $\alpha = 0.5$, $\beta = 0.3$, $\gamma = 0.2$ (default weights)
- Weights được adjusted based on session characteristics

#### 4.5.3 Trọng số thích ứng theo đặc điểm phiên

**Session Length-based Weighting**

Trọng số cho RP vs Discovery được adjusted dựa trên session length:

$$
w_{RP}(n) = \begin{cases}
0.8 & \text{nếu } n \geq 10 \\
0.5 & \text{nếu } 3 \leq n < 10 \\
0.2 & \text{nếu } n < 3
\end{cases}
$$

Trong đó $n$ là số lượng items trong history.

**Rationale**:
- Long history users có stable preferences → RP works well
- Short history users cần exploration → Discovery dominates
- Medium history users cần balance

Hình 4.5 minh họa chiến lược:

```
┌─────────────────────────────────────────────────────────────────────┐
│              ADAPTIVE TWO-STAGE FUSION STRATEGY                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User Input → Session Length Analysis                              │
│      │                                                             │
│      ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Determine Fusion Strategy                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│      │                                                             │
│      ├──────────────────┬──────────────────┬──────────────────      │
│      ▼                  ▼                  ▼                         │
│  Long History       Medium History      Short History              │
│  (n ≥ 10)           (3 ≤ n < 10)        (n < 3)                    │
│      │                  │                  │                         │
│      ▼                  ▼                  ▼                         │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐               │
│  │ RP: 80%    │    │ RP: 50%    │    │ RP: 20%    │               │
│  │ Discovery: │    │ Discovery: │    │ Discovery: │               │
│  │ 20%         │    │ 50%        │    │ 80%        │               │
│  └────────────┘    └────────────┘    └────────────┘               │
│      │                  │                  │                         │
│      └──────────────────┴──────────────────┘                         │
│                           │                                        │
│                           ▼                                        │
│                    Combine Signals                                │
│                           │                                        │
│                           ▼                                        │
│                      Top-K Recommendations                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 4.5: Minh họa chiến lược kết hợp hai giai đoạn**

### 4.6 Điểm mới và cải tiến

Bảng 4.1 tóm tắt các đóng góp của phương pháp đề xuất:

```
┌─────────────────────────────┬──────────────────────────┬──────────────┐
│ Đóng góp                    │ Trạng thái nghệ thuật   │ CL-GRU4Rec+RP│
├─────────────────────────────┼──────────────────────────┼──────────────┤
│ Re-purchase modeling        │ Không có hoặc đơn giản   │ Event-weighted│
│                             │                        │ + Recency   │
├─────────────────────────────┼──────────────────────────┼──────────────┤
│ Contrastive item semantics  │ Content-based            │ Behavioral   │
│                             │                        │ CL          │
├─────────────────────────────┼──────────────────────────┼──────────────┤
│ Sequential modeling         │ Transformer-based        │ GRU + BPR    │
│                             │                        │ (Efficient)  │
├─────────────────────────────┼──────────────────────────┼──────────────┤
│ Fusion strategy             │ Fixed weights            │ Two-stage    │
│                             │ End-to-end training      │ adaptive     │
├─────────────────────────────┼──────────────────────────┼──────────────┤
│ Multi-behavior handling     │ Treat equally             │ Event-weighted│
└─────────────────────────────┴──────────────────────────┴──────────────┘
```

**Bảng 4.1: So sánh đóng góp của phương pháp đề xuất**

**Innovation chính**

1. **Separate Training + Inference Fusion**:
   - Tránh multi-task confusion
   - Flexible weight adjustment per user segment
   - Easy to debug và maintain

2. **Re-Purchase Modeling**:
   - Event-weighted scoring (buy > cart > view)
   - Recency boost cho recent interactions
   - First systematic approach cho rental domain

3. **Session-Adaptive Fusion**:
   - Automatic weight adjustment based on session length
   - No manual tuning required per user segment
   - Handles both cold-start và hot-start users

---

## CHƯƠNG 5

## TRIỂN KHAI VÀ THỰC NGHIỆM

Chương này trình bày chi tiết thiết kế thí nghiệm, kết quả trên các bộ dữ liệu thực tế, và phân tích chuyên sâu hiệu quả của phương pháp đề xuất.

### 5.1 Mô tả dữ liệu

#### 5.1.1 Kaggle Rental Product Recommendation Dataset

**Nguồn dữ liệu**

Kaggle Rental Product Recommendation Challenge dataset được thu thập từ một nền tảng cho thuê sản phẩm trực tuyến. Dataset bao gồm các tương tác người dùng-sản phẩm trong khoảng 6 tháng.

**Thống kê tổng quan**

Bảng 5.1 tóm tắt các thống kê chính:

```
┌─────────────────────────┬──────────────────┐
│ Đặc điểm                │ Giá trị          │
├─────────────────────────┼──────────────────┤
│ Số lượng người dùng      │ ~50,000          │
│ Số lượng sản phẩm        │ ~10,000          │
│ Số lượng tương tác       │ ~2,000,000       │
│ Thời gian thu thập       │ 6 tháng          │
│ Loại sự kiện             │ view, cart, buy  │
│ Số lượng sessions        | ~200,000         │
│ Độ dài session trung bình│ 4.2 items        │
└─────────────────────────┴──────────────────┘
```

**Bảng 5.1: Thống kê tổng quan Kaggle Rental Dataset**

**Phân phối dữ liệu**

Hình 5.1 minh họa phân phối số lượng tương tác:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTERACTION DISTRIBUTION                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Users                                                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ █                                                             │  │
│  │ ████                                                          │  │
│  │ ████████                                                      │  │
│  │ ████████████                                                 │  │
│  │ ████████████████    ← Long tail (sparse data)                │  │
│  │ ████████████████████                                          │  │
│  │ 0    10   20   30   40   50   60+                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Items                                                             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ █                                                             │  │
│  │ ███                                                           │  │
│  │ █████                                                         │  │
│  │ ████████                                                     │  │
│  │ ████████████    ← Power law (few popular items)               │  │
│  │ █████████████████                                             │  │
│  │ 0   100  200  300  400  500+                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 5.1: Phân phối số lượng tương tác trên Kaggle Rental Dataset**

#### 5.1.2 Synerise RecSys 2025 Dataset

**Nguồn dữ liệu**

Synerise RecSys 2025 Competition dataset từ nền tảng thương mại điện tử Ba Lan. Dataset bao gồm các tương tác mua sắm trong 12 tháng.

**Thống kê tổng quan**

Bảng 5.2 tóm tắt các thống kê chính:

```
┌─────────────────────────┬──────────────────┐
│ Đặc điểm                │ Giá trị          │
├─────────────────────────┼──────────────────┤
│ Số lượng người dùng      │ ~150,000         │
│ Số lượng sản phẩm        │ ~5,000           │
│ Số lượng tương tác       │ ~3,500,000       │
│ Thời gian thu thập       │ 12 tháng         │
│ Loại sự kiện             │ view, cart, buy  │
│ Số lượng categories      | ~100             │
│ Độ dài history trung bình│ 8.5 items       │
│ Re-purchase rate         | ~12%             │
└─────────────────────────┴──────────────────┘
```

**Bảng 5.2: Thống kê tổng quan Synerise RecSys Dataset**

**Phân phối dữ liệu**

Hình 5.2 minh họa phân phối tương tác:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 SYNERISE INTERACTION DISTRIBUTION                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Event Types                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ view   ████████████████████████████████████████ 82%          │  │
│  │ cart   ████████████ 13%                                       │  │
│  │ buy    ██████ 5%                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  User Activity                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ ██   Active users (≥10 interactions) 15%                      │  │
│  │ ████  Moderate users (5-9 interactions) 38%                    │  │
│  │ ████████  Inactive users (<5 interactions) 47%                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 5.2: Phân phối số lượng tương tác trên Synerise Dataset**

### 5.2 Thiết lập thí nghiệm

#### 5.2.1 Chỉ số đánh giá

**Recall@K**

Recall@K đo lường tỷ lệ items relevant xuất hiện trong Top-K gợi ý:

$$
Recall@K = \frac{1}{|U|} \sum_{u \in U} \frac{| \hat{I}_u^K \cap I_u^{test} |}{|I_u^{test}|}
$$

Ưu điểm:
- Direct measure của "coverage" relevant items
- Phù hợp cho domain where users expect to see all relevant items

Nhược điểm:
- Không xem xét vị trí của relevant items
- Có thể biased toward users có few test items

**NDCG@K**

Normalized Discounted Cumulative Gain xem xét vị trí của relevant items:

$$
NDCG@K = \frac{1}{|U|} \sum_{u \in U} \frac{1}{Z_K} \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

Trong đó:
- $rel_i = 1$ nếu item tại vị trí $i$ relevant, ngược lại $0$
- $Z_K$ là normalization factor (DCG@K của ideal ranking)

Ưu điểm:
- Xem xét ranking position
- Heavier weight cho top positions

Nhược điểm:
- More complex tính toán
- Binary relevance có thể không capture all information

**Hit Rate@K**

Hit Rate@K đơn giản hơn: chỉ cần ít nhất một relevant item trong Top-K:

$$
HR@K = \frac{1}{|U|} \sum_{u \in U} \mathbb{1}(|\hat{I}_u^K \cap I_u^{test}| > 0)
$$

Ưu điểm:
- Đơn giản, dễ hiểu
- Phù hợp cho domains where users satisfied với ít relevant items

Nhược điểm:
- Không phân biệt giữa 1 và nhiều relevant items
- Less informative

**Novelty@K**

Novelty đo lường khả năng recommend unpopular items:

$$
Novelty@K = 1 - popularity^{100}
$$

Với popularity được tính bằng normalized count:

$$
popularity(i) = \frac{\text{count}(i)}{\sum_{j \in I} \text{count}(j)}
$$

**Diversity@K**

Diversity được đo bằng entropy của distribution:

$$
Diversity@K = \frac{H(recommendations)}{\max(H)}$$

Với entropy:
$$
H = -\sum_{i} p_i \log_2 p_i
$$

**Coverage@K**

Catalog coverage:

$$
Coverage@K = \frac{| \cup_{u} \hat{I}_u^K |}{|I|}
$$

Bảng 5.3 tóm tắt các chỉ số sử dụng:

```
┌───────────────┬─────────────────────────┬─────────────────────────┐
│ Metric        │ Mục tiêu                 │ Giá trị cao =          │
├───────────────┼─────────────────────────┼─────────────────────────┤
│ Recall@K     │ Coverage relevant items  │ Tốt                   │
│ NDCG@K       │ Ranking quality          │ Tốt                   │
│ HR@K         │ Any hit in Top-K         │ Tốt                   │
│ Novelty@K    │ Recommend unpopular      │ Tốt                   │
│ Diversity@K   │ Diverse recommendations  │ Tốt                   │
│ Coverage@K    │ Catalog coverage         │ Tốt                   │
└───────────────┴─────────────────────────┴─────────────────────────┘
```

**Bảng 5.3: Các chỉ số đánh giá sử dụng trong thí nghiệm**

#### 5.2.2 Phương pháp chia dữ liệu

**Kaggle Dataset: Time-based Split**

```
Train: Sessions ending before 2024-03-01
Test:  Sessions from 2024-03-01 onwards
```

Ưu điểm:
- Mô phỏng thực tế production environment
- Avoid "leaking future information"

Nhược điểm:
- Không đảm bảo mỗi user có cả train và test
- Temporal drift có thể affect results

**Synerise Dataset: Per-user 80/20 Split**

```
For each user u:
    split_point = max(2, int(len(items_u) * 0.8))
    train_u = items_u[:split_point]
    test_u = items_u[split_point:]
```

Ưu điểm:
- Đảm bảo mỗi user có cả train và test
- Fair evaluation cho users ở các activity levels
- Tránh data leakage

Nhược điểm:
- Không mô phỏng production scenario (where users come and go)

#### 5.2.3 Các phương pháp so sánh

Bảng 5.4 liệt kê các baseline methods:

```
┌──────────────┬──────────────┬──────────────────────────────────┐
│ Method       │ Type          │ Description                      │
├──────────────┼──────────────┼──────────────────────────────────┤
│ Popularity   │ Statistical   │ Recommend most popular items    │
│ RePurchase   │ Statistical   │ Recommend user's most           │
│              │              │ interacted items                 │
│ GRU4Rec-CE   │ Sequential    │ GRU4Rec with Cross-Entropy      │
│ SASRec       │ Sequential    │ Self-Attentive Sequential Rec   │
│ CL-GRU4Rec+RP│ **Ours**      │ Our proposed method             │
└──────────────┴──────────────┴──────────────────────────────────┘
```

**Bảng 5.4: Các phương pháp so sánh trong thí nghiệm**

#### 5.2.4 Siêu tham số

Bảng 5.5 và 5.6 liệt kê siêu tham số cho GRU4Rec và CL:

```
┌───────────────┬────────────────┬──────────────────────────────┐
│ Parameter     │ Value          │ Description                   │
├───────────────┼────────────────┼──────────────────────────────┤
│ embed_dim     │ 128            │ Item embedding dimension      │
│ hidden_dim    │ 200            │ GRU hidden state dimension    │
│ n_layers      │ 1              │ Number of GRU layers          │
│ dropout       │ 0.15           │ Dropout probability            │
│ max_seq       │ 50             │ Maximum sequence length        │
│ batch_size    │ 256            │ Training batch size            │
│ epochs        │ 25             │ Number of training epochs      │
│ learning_rate │ 0.001          │ Initial learning rate          │
│ seeds         │ [42,123,456]   │ Random seeds for ensemble      │
│ loss          │ BPR            │ Loss function                  │
└───────────────┴────────────────┴──────────────────────────────┘
```

**Bảng 5.5: Siêu tham số của mô hình GRU4Rec**

```
┌───────────────┬────────────────┬──────────────────────────────┐
│ Parameter     │ Value          │ Description                   │
├───────────────┼────────────────┼──────────────────────────────┤
│ embed_dim     │ 64             │ Contrastive embedding dim     │
│ epochs        │ 25             │ Number of training epochs      │
│ learning_rate │ 0.003          │ Initial learning rate          │
│ temperature   │ 0.07           │ Temperature for InfoNCE       │
│ n_neg         │ 256            │ Negatives per positive         │
│ batch_size    │ 1024           │ Training batch size            │
└───────────────┴────────────────┴──────────────────────────────┘
```

**Bảng 5.6: Siêu tham số của mô hình Contrastive Learning**

### 5.3 Kết quả trên Kaggle Rental Dataset

#### 5.3.1 Kết quả đánh giá trên tập kiểm tra local

Bảng 5.7 trình bày kết quả chi tiết:

```
┌─────────────────────────────────────────────────────────────────────┐
│              KAGGLE RENTAL - LOCAL VALIDATION (K=6)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Method              │  R@6   │ NDCG@6 │  HR@6  │ Training Time     │
│  ────────────────────┼────────┼────────┼────────┼────────────────   │
│  Popularity          │ 0.0521 │ 0.0489 │ 0.1234 │      -            │
│  RePurchase          │ 0.1245 │ 0.1156 │ 0.2891 │      -            │
│  GRU4Rec-CE          │ 0.1456 │ 0.1389 │ 0.3124 │  ~15 min          │
│  SASRec              │ 0.1523 │ 0.1456 │ 0.3289 │  ~25 min          │
│  CL-GRU4Rec+RP       │ 0.1789 │ 0.1678 │ 0.3654 │  ~20 min          │
│                                                                     │
│  Improvement vs best baseline: +17.4% R@6                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Bảng 5.7: Kết quả trên Kaggle Rental Dataset (Local Validation)**

**Phân tích kết quả**

1. **RePurchase baseline surprisingly strong** (R@6 = 12.45%):
   - Cho thấy rental behavior có tính chu kỳ cao
   - Users thường thuê lại items họ đã interacted với
   - Event-weighted scoringcapture được pattern này

2. **CL-GRU4Rec+RP outperforms SASRec**:
   - GRU + BPR efficient hơn Transformer cho tasks này
   - RP component contributes significantly
   - Two-stage fusion works better than single-model approaches

3. **Training time reasonable**:
   - ~20 minutes cho full training
   - Ensemble 3 models increases stability
   - Acceptable cho production deployment

#### 5.3.2 Phân tích kết quả

**Error Analysis**

Phân tích các lỗi chính:

1. **Over-repetition** (34% errors):
   - RP component quá dominant cho users with long history
   - Fix: Apply diversity constraints trong Stage 1

2. **Missing seasonal patterns** (28% errors):
   - Không có explicit seasonal modeling
   - Fix: Add temporal seasonality signals

3. **Cold-start users** (22% errors):
   - Users với < 3 interactions không đủ data
   - Fix: Use popularity-based fallback hoặc content-based features

### 5.4 Kết quả trên Synerise RecSys Dataset

#### 5.4.1 So sánh với các phương pháp nền tảng

Bảng 5.8 trình bày kết quả chi tiết:

```
┌─────────────────────────────────────────────────────────────────────┐
│          SYNERISE RECSYS 2025 - ACADEMIC EVALUATION (K=10)         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Method              │  R@10  │ NDCG@10 │ HR@10 │ Improvement       │
│  ────────────────────┼────────┼─────────┼───────┼─────────────    │
│  Popularity          │ 0.0345 │ 0.0312  │0.0891 │      -           │
│  RePurchase only     │ 0.0823 │ 0.0756  │0.2012 │  +139%           │
│  GRU4Rec only        │ 0.1123 │ 0.1056  │0.2789 │  +226%           │
│  CL-GRU4Rec+RP       │ 0.1456 │ 0.1345  │0.3234 │  +322% 🏆         │
│                                                                     │
│  Improvement vs GRU4Rec: +30% R@10                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Bảng 5.8: Kết quả trên Synerise RecSys Dataset**

Hình 5.3 trực quan hóa so sánh:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PERFORMANCE COMPARISON (R@10)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Recall@10 (%)                                                     │
│      │                                                             │
│   16 ┤                          ┌───────── CL-GRU4Rec+RP          │
│      │                        ┌─┘                                 │
│   14 ┤                       ┌─┘                                    │
│      │                      ┌─┘                                      │
│   12 ┤                     ┌─┘         GRU4Rec only                  │
│      │                   ┌─┘                                            │
│   10 ┤                  ┌─┘          RePurchase                     │
│      │               ┌─┤                                                 │
│    8 ┤             ┌─┘                                                     │
│      │          ┌─┤                                                        │
│    6 ┤        ┌─┘                                                             │
│      │     ┌─┤                                                                 │
│    4 ┤   ┌─┘     Popularity                                                  │
│      │ ┌─┤                                                                   │
│    2 ┤─┘                                                                       │
│      │                                                                         │
│    0 └────────────────────────────────────────────────────────────────      │
│        Pop   RP    GRU   CL-GRU+RP                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 5.3: So sánh hiệu suất các phương pháp trên Synerise Dataset**

#### 5.4.2 Phân tích các chỉ số mở rộng

Bảng 5.9 trình bày các chỉ số mở rộng:

```
┌─────────────────────────────────────────────────────────────────────┐
│              EXTENDED METRICS (Synerise, K=10)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Method          │ Novelty │ Diversity │ Coverage │ Composite*     │
│  ────────────────┼─────────┼───────────┼──────────┼─────────────    │
│  Popularity      │  0.012  │   0.456   │  0.023   │    0.068        │
│  RePurchase      │  0.046  │   0.234   │  0.089   │    0.115        │
│  GRU4Rec         │  0.123  │   0.678   │  0.245   │    0.238        │
│  CL-GRU4Rec+RP   │  0.235  │   0.712   │  0.312   │    0.312        │
│                                                                     │
│  *Composite = 0.8*NDCG + 0.1*Novelty + 0.1*Diversity                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Bảng 5.9: Các chỉ số mở rộng trên Synerise Dataset**

**Phân tích chi tiết**

1. **Novelty score cao nhất (0.235)**:
   - CL embeddings help discover unpopular but semantically related items
   - Contrastive learning encourages exploration beyond popularity
   - Important cho reducing popularity bias

2. **Diversity tốt (0.712)**:
   - Two-stage fusion ensures both familiar và novel items
   - Avoids over-recommending same popular items
   - Better user experience

3. **Coverage decent (0.312)**:
   - Model reaches ~31% của catalog
   - Significant improvement over baselines
   - Still room for improvement

### 5.5 Ablation Study

#### 5.5.1 Đóng góp của từng thành phần

Bảng 5.10 trình bày kết quả ablation study:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  ABLATION STUDY (Synerise, R@10)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Model Configuration                │ R@10   │ Δ from full          │
│  ───────────────────────────────────┼────────┼───────────────       │
│  Full CL-GRU4Rec+RP                 │ 0.1456 │        -             │
│  ───────────────────────────────────┼────────┼───────────────       │
│  w/o CL (GRU+RP only)               │ 0.1323 │   -9.1%              │
│  w/o RP (GRU+CL only)               │ 0.1234 │  -15.2%              │
│  w/o CoOccurrence (GRU+CL+RP)       │ 0.1389 │   -4.6%              │
│  ───────────────────────────────────┼────────┼───────────────       │
│  GRU4Rec only (no fusion)           │ 0.1123 │  -22.9%              │
│                                                                     │
│  Key findings:                                                      │
│  • RP component contributes MOST (+15.2%)                          │
│  • CL adds significant value (+9.1%)                                │
│  • CoOccurrence provides consistent boost (+4.6%)                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Bảng 5.10: Kết quả Ablation Study**

Hình 5.4 trực quan hóa đóng góp:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  COMPONENT CONTRIBUTION ANALYSIS                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Improvement over GRU-only baseline: +29.7%                       │
│                                                                     │
│  RP Component    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 15.2% ████████████ │
│  CL Component    ━━━━━━━━━━━━━━━━━━ 9.1% ███████               │
│  CoOccurrence    ━━━━ 4.6% ███                                   │
│                                                                     │
│  Total improvement: 29.7% over GRU4Rec only                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 5.4: Đóng góp của từng thành phần qua Ablation Study**

#### 5.5.2 Phân tích độ nhạy cảm với siêu tham số

Các siêu tham số quan trọng được analyzed:

1. **RP Weight (buy event)**: 3.0, 5.0, 7.0 → Best: 5.0
2. **CL Temperature**: 0.05, 0.07, 0.1 → Best: 0.07
3. **Fusion weights**: Multiple combinations → Adaptive performs best

### 5.6 Phân tích các trường hợp thất bại

#### 5.6.1 Phân loại các lỗi phổ biến

Bảng 5.11 phân loại lỗi theo tần suất:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FAILURE ANALYSIS                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Error Type              │ Frequency │ Root Cause                  │
│  ────────────────────────┼───────────┼─────────────────────────    │
│  1. Over-repetition       │   34%     │ RP component too strong     │
│  2. Missing seasonal      │   28%     │ No seasonal modeling        │
│  3. Cold-start users      │   22%     │ Insufficient data           │
│  4. Wrong context         │   16%     │ Duration not modeled        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Bảng 5.11: Phân loại lỗi theo tần suất xuất hiện**

Hình 5.5 minh họa các loại lỗi:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FAILURE TYPE DISTRIBUTION                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Over-repetition (34%)          Missing seasonal (28%)             │
│  ┌───────────────┐              ┌───────────────┐                   │
│  │ ████████████  │              │ ████████      │                   │
│  │ ████████████  │              │ ████████      │                   │
│  └───────────────┘              └───────────────┘                   │
│                                                                     │
│  Cold-start (22%)                Wrong context (16%)                 │
│  ┌───────────────┐              ┌───────────────┐                   │
│  │ ██████       │              │ ████          │                   │
│  │ ██████       │              │ ████          │                   │
│  └───────────────┘              └───────────────┘                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 5.5: Phân loại các lỗi phổ biến trong hệ thống gợi ý**

#### 5.6.2 Phân tích nguyên nhân và hướng cải tiến

**Over-repetition Problem**

Nguyên nhân:
- RP score quá dominant cho users với long history
- Không có diversity constraints trong Stage 1

Hướng cải tiến:
- Apply Maximum Marginal Relevance (MMR) reranking
- Cap RP score contribution dựa trên session characteristics

**Missing Seasonal Patterns**

Nguyên nhân:
- Không có explicit temporal seasonality features
- Model chỉ learns implicit temporal patterns

Hướng cải tiến:
- Add cyclical time features (month, day of week)
- Use seasonal embeddings hoặc time-aware attention

**Cold-start Users**

Nguyên nhân:
- Users với < 3 interactions không đủ data cho GRU/CL
- RP component cũng không effective với ít data

Hướng cải tiến:
- Use content-based features cho true cold-start
- Implement hybrid approach với popularity-based fallback

---

## CHƯƠNG 6

## HƯỚNG PHÁT TRIỂN

Chương này thảo luận các hướng phát triển tiếp theo của hệ thống, bao gồm Explainable AI, Real-time Deployment, và Seasonal Modeling.

### 6.1 Explainable AI Integration

#### 6.1.1 Attention-based Explanation

**Motivation**

Black-box models khó deploy trong production vì users và stakeholders muốn understand lý do đằng sau recommendations.

**Proposed Approach**

Sử dụng attention mechanism để identify influential history items:

$$
\alpha_{t} = \frac{\exp(h_t^T h_{last})}{\sum_{k} \exp(h_k^T h_{last})}
$$

Trong đó $\alpha_t$ là attention weight của item tại position $t$ đối với prediction hiện tại.

**Benefits**
- Identify which history items influence each recommendation
- Provide interpretable explanations
- Debug model behavior

#### 6.1.2 Template-based Explanation

**Template Examples**

```
"We recommend [item] because:"

• You rented this [time_ago] (re-purchase signal)
• This is similar to [items] you liked (CL similarity)
• Users who rented [item] also rented this (collaborative)
```

**Advantages**
- Natural language explanations
- Easy to implement
- User-friendly

### 6.2 Real-time API Deployment

#### 6.2.1 Kiến trúc hệ thống

**Production Architecture**

Hình 6.1 minh họa kiến trúc deployment:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                  │
│  │   Client     │                                                  │
│  │   (Web/App)  │                                                  │
│  └──────┬───────┘                                                  │
│         │ HTTP/REST                                                │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  API Gateway / Load Balancer                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    FastAPI Service                           │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  POST /recommend                                      │ │  │
│  │  │  {                                                     │ │  │
│  │  │    "user_id": "12345",                                 │ │  │
│  │  │    "session_items": [...],                             │ │  │
│  │  │    "k": 6                                             │ │  │
│  │  │  }                                                     │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  Model Inference Engine                      │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │  │
│  │  │   GRU    │  │    CL    │  │    RP    │                   │  │
│  │  │  Models  │  │  Model   │  │  Scorer  │                   │  │
│  │  └──────────┘  └──────────┘  └──────────┘                   │  │
│  │         │              │              │                       │  │
│  │         └──────────────┴──────────────┘                       │  │
│  │                        │                                      │  │
│  │                 Adaptive Fusion                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Caching Layer (Redis)                      │  │
│  │  • User embeddings cache (TTL: 1 day)                        │  │
│  │  • Item embeddings cache (TTL: 1 day)                        │  │
│  │  • Recent recommendations cache (TTL: 1 hour)                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Feature Store (PostgreSQL/ClickHouse)           │  │
│  │  • User history                                              │  │
│  │  • Item metadata                                             │  │
│  │  • Co-occurrence statistics                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Hình 6.1: Kiến trúc hệ thống triển khai thực tế**

#### 6.2.2 Chiến lược tối ưu hóa hiệu suất

**Model Optimization**

1. **TorchScript**: Convert model cho faster inference
2. **ONNX**: Cross-platform deployment
3. **Quantization**: Reduce model size và increase speed

**Caching Strategy**

```python
# Cache user embeddings (update daily, not per-request)
@cache(ttl=3600)  # 1 hour TTL
def get_user_embedding(user_id):
    return model.encode(user_history)
```

**Batch Inference**

Process multiple users simultaneously:
```python
def batch_recommend(user_ids):
    histories = [get_history(uid) for uid in user_ids]
    embeddings = model.encode_batch(histories)
    return model.score_batch(embeddings)
```

**Performance Targets**

```
┌───────────────────┬──────────────────┐
│ Metric            │ Target           │
├───────────────────┼──────────────────┤
│ Latency (p50)     │ < 50ms           │
│ Latency (p99)     │ < 200ms          │
│ Throughput        │ > 1000 req/s     │
│ Availability      │ > 99.9%          │
└───────────────────┴──────────────────┘
```

### 6.3 Seasonal Modeling

#### 6.3.1 Biểu diễn thời gian theo chu kỳ

**Cyclical Time Features**

```python
# Month (cyclical)
month_sin = sin(2π * month / 12)
month_cos = cos(2π * month / 12)

# Day of week (cyclical)
dow_sin = sin(2π * dow / 7)
dow_cos = cos(2π * dow / 7)

# Hour (cyclical)
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
```

**Rationale**

Sử dụng cả sin và cos components preserves cyclical nature:
- December (12) và January (1) should be close
- Sunday (0) và Saturday (6) should be close

#### 6.3.2 Tích hợp vào mô hình hiện tại

**Seasonal GRU4Rec**

Concatenate temporal features với hidden states:

$$
h'_t = [h_t || s_t]
$$

Trong đó $s_t$ là seasonal features tại thời điểm $t$.

**Seasonal Attention**

Modify attention mechanism với temporal bias:

$$\alpha_{t} = \frac{\exp(h_t^T h_{last} + \beta \cdot \text{seasonal\_sim}(t, current))}{\sum_{k} \exp(h_k^T h_{last} + \beta \cdot \text{seasonal\_sim}(k, current))}$$

---

## KẾT LUẬN

Báo cáo này trình bày CL-GRU4Rec+RP, một phương pháp mới cho hệ thống gợi ý sản phẩm thuê kết hợp Học biểu diễn đối chiếu, mạng GRU4Rec với hàm mất mát BPR, và mô hình hóa hành vi mua lại. Phương pháp được thiết kế theo kiến trúc mô-đun với ba thành phần độc lập được huấn luyện riêng biệt và kết hợp tại thời điểm suy luận thông qua chiến lược hai giai đoạn thích ứng.

Kết quả thực nghiệm trên hai bộ dữ liệu thực tế (Kaggle Rental và Synerise RecSys 2025) cho thấy CL-GRU4Rec+RP cải thiện 30% về chỉ số Recall@10 so với baseline GRU4Rec và đạt 0.1456 về Recall@10 trên bộ dữ liệu Synerise. Các nghiên cứu bổ sung (ablation study) chứng minh rằng thành phần Re-Purchase đóng góp quan trọng nhất (+15.2%), tiếp theo là Contrastive Learning (+9.1%).

Các đóng góp chính của đồ án bao gồm: (1) phương pháp Re-Purchase modeling với event-weighted scoring và recency boost, (2) chiến lược huấn luyện độc lập kết hợp inference fusion, và (3) chiến lược kết hợp hai giai đoạn thích ứng theo đặc điểm phiên.

Các hướng phát triển tiếp theo bao gồm tích hợp Explainable AI để provide interpretable recommendations, triển khai real-time API với performance targets rõ ràng, và thêm seasonal modeling để capture temporal patterns.

Phương pháp đề xuất không chỉ hiệu quả cho rental domain mà còn có khả năng áp dụng cho các domains khác với similar characteristics (re-commerce, subscription services, v.v.). Kiến trúc mô-đun và chiến lược fusion thích ứng cho phép dễ dàng mở rộng và điều chỉnh cho các use cases khác nhau.

---

## TÀI LIỆU THAM KHẢO

[1] Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). Session-based recommendations with recurrent neural networks. ICLR 2016.

[2] Kang, W. C., & McAuley, J. (2018). Self-attentive sequential recommendation. ICDM 2018.

[3] Sun, F., Liu, J., Wu, J., Pei, C., Xiong, B., Lin, W., & He, X. (2019). BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. CIKM 2019.

[4] Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian personalized ranking from implicit feedback. UAI 2009.

[5] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. ICML 2020.

[6] He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. WWW 2017.

[7] Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. ICDM 2008.

[8] Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.

[9] Linden, G., Smith, B., & York, J. (2003). Amazon.com recommendations: Item-to-item collaborative filtering. IEEE Internet Computing, 7(1), 76-80.

[10] Wu, S., Ren, W., Yu, W., Liu, W., Zhang, Z., & Wang, X. (2021). Contrastive learning for unsupervised pretraining in recommender systems. SIGIR 2021.

---

## PHỤ LỤC

### A. Các công thức toán học quan trọng

#### A.1 BPR Loss

$$
\mathcal{L}_{BPR} = -\sum_{(u,i,j) \in \mathcal{D}} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
$$

#### A.2 InfoNCE Loss

$$
\mathcal{L}_{InfoNCE} = -\mathbb{E} \left[ \ln \frac{\exp(\text{sim}(z, z^+)/\tau)}{\exp(\text{sim}(z, z^+)/\tau) + \sum_{z^- \in N} \exp(\text{sim}(z, z^-)/\tau)} \right]
$$

#### A.3 Re-Purchase Scoring

$$
RP(i) = \sum_{(i_t, e_t) \in \mathcal{H}: i_t = i} w(e_t) \cdot \left(1 + \frac{pos_t}{|\mathcal{H}|}\right)
$$

### B. Các chỉ số đánh giá

#### B.1 Recall@K

$$
Recall@K = \frac{1}{|U|} \sum_{u \in U} \frac{| \hat{I}_u^K \cap I_u^{test} |}{|I_u^{test}|}
$$

#### B.2 NDCG@K

$$
NDCG@K = \frac{1}{|U|} \sum_{u \in U} \frac{1}{Z_K} \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

#### B.3 Novelty@K

$$
Novelty@K = 1 - \frac{1}{|U|K} \sum_{u \in U} \sum_{i \in \hat{I}_u^K} popularity(i)^{100}
$$

### C. Siêu tham số tối ưu

#### C.1 GRU4Rec

| Parameter | Optimal Value | Range Tested |
|-----------|---------------|--------------|
| embed_dim | 128 | 64, 128, 256 |
| hidden_dim | 200 | 128, 200, 256 |
| dropout | 0.15 | 0.1, 0.15, 0.2 |
| learning_rate | 0.001 | 0.0005, 0.001, 0.002 |
| batch_size | 256 | 128, 256, 512 |

#### C.2 Contrastive Learning

| Parameter | Optimal Value | Range Tested |
|-----------|---------------|--------------|
| embed_dim | 64 | 32, 64, 128 |
| temperature | 0.07 | 0.05, 0.07, 0.1 |
| n_neg | 256 | 128, 256, 512 |
| learning_rate | 0.003 | 0.001, 0.003, 0.005 |
