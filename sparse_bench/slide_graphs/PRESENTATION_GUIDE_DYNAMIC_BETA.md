# Kịch bản trình bày CEARF-N Dynamic-\(\beta\)

## 1. Thông điệp trung tâm

**Một câu cần để reviewer nhớ:**

> CEARF-N không dùng validation để chọn một trọng số fusion toàn cục. Phương pháp học một luật phân bổ neural liên tục theo từng truy vấn từ rank out-of-fit của tập train, rồi giới hạn mỗi hiệu chỉnh quanh prior toàn cục trong \(\pm 0.10\).

Tên đầy đủ:

> **CEARF-N — Contextual Expert-Allocation Rank Fusion with a
> Neural residual**

Đối tượng được đề xuất là **bounded expert-disagreement correction** — một
conditional stacker trên các frozen ranker — không phải sequence encoder mới:

\[
F_q(x)=(1-\beta_q)\,\mathrm{RR}_M(x)+\beta_q\,\mathrm{RR}_N(x),
\]

\[
\beta_q=\beta_{\mathrm{OOF}}+
\Delta_{\mathrm{eff}}\tanh(\mathbf w^\top\widetilde{\mathbf z}_q+b),
\qquad \Delta\leq 0.10.
\]

Ba feature target-free của gate:

1. \(\log(1+\text{context length})\);
2. \(\log(1+\text{last-item frequency})\);
3. tail indicator của last item.

Toàn bộ allocator chỉ có **năm scalar học được**: một global prior
\(\beta_{\mathrm{OOF}}\), ba coefficient và một bias.

### Điều nên khẳng định

- Fusion của hai nguồn bằng chứng bổ sung nhau rõ rệt trên cả ba domain.
- Dynamic allocation được học từ training-only OOF ranks, không grid-search
  \(\beta\) trên validation.
- Hiệu chỉnh bị chặn trong cả allocation space và fused-score space.
- Rank-only expert interface bất biến với mọi strictly monotone raw-score transformation giữ
  nguyên top-120 ordering.
- Video Games cho bằng chứng rõ nhất rằng gate sắp xếp được các query có
  neural advantage khác nhau.
- Khi giữ nguyên beta multiset, learned query assignment cải thiện nDCG trên
  Video và có tín hiệu hỗ trợ ở Diginetica nDCG@10.
- Dynamic post-processing rất nhỏ so với expert retrieval.

### Điều không được khẳng định

- Không nói CEARF-N là universal SOTA.
- Không nói phần tăng lớn so với endpoint đều đến từ dynamic gate; phần lớn
  đến từ **fusion complementarity**.
- Không nói coefficient của gate là causal effect.
- Không nói so sánh external baseline trên Amazon là metadata-matched.
- Không nói Baby/Diginetica có dynamic-over-global effect được xác nhận nếu
  khoảng tin cậy cuối cùng vẫn cắt 0.
- Dùng NARM expert swap như sensitivity control, không đổi primary headline.

---

## 2. Trạng thái bằng chứng và quy tắc dùng chart

### Đã có thể dùng

- `00-method-architecture.png`: hai lane OOF training và target-free
  inference của phương pháp.
- `00-protocol-lineage.png`: thứ tự khai báo validation, OOF profile/gate,
  freeze manifest, refit expert và test.
- `14-fusion-ablation-recall20.png`: endpoint complementarity, đúng ba seed
  42/123/456.
- `01-allocation-modes-utility.png`: chuỗi endpoint \(\rightarrow\) equal
  mixing \(\rightarrow\) global/coarse/dynamic allocation.
- `03-rescue-damage-vs-memory.png`: thay đổi ở mức query so với memory.
- `04-dynamic-beta-distribution.png`: phân phối \(\beta_q\).
- `05-dynamic-beta-context-behavior.png`: hành vi allocation theo ba feature.
- `06-rrf-k-sensitivity.png`: \(k\in\{10,20,60\}\).
- `07-inference-performance.png`: timing của allocation và RRF.
- `08-external-baseline-comparison-recall20.png`: system-level comparison,
  kèm disclosure metadata.
- `09-expert-swap-pasgr-vs-narm.png`: sensitivity khi thay PASGR bằng
  ID-only NARM trong cùng allocator.
- `12-beta-decile-mechanism-ndcg20.png`: chỉ dùng bản được sinh từ summary
  hoàn chỉnh 3-domain × 3-seed.
- `13-dynamic-vs-global-six-metric-paired-delta.png`: chỉ dùng bản được sinh
  từ summary hoàn chỉnh 3-domain × 3-seed.
- `11-allocation-capacity-controls.png`: allocation family hoàn chỉnh 3×3.
- `15-beta-assignment-paired-effects.png`: paired same-multiset assignment
  control hoàn chỉnh 3×3.
- `16-primary-gate-standardized-coefficients.png`: coefficient chính xác của
  frozen primary gates.
- `10-fusion-operator-rrf-vs-combsum.png`: operator control hoàn chỉnh 3×3.
- `17-bounded-pair-certificate.png`: sơ đồ formal certificate, không chứa số
  thực nghiệm.

**Release gate:** trước khi xuất slide, mở `chart_manifest.json`; chart nào có
trạng thái `skipped`, seed thiếu, hoặc source SHA không trùng artifact cuối thì
không được đưa vào deck.

---

## 3. Bộ RQ dùng khi trình bày

### RQ1 — Hai expert có bổ sung bằng chứng cho nhau không, và dynamic allocation thêm gì ngoài equal/global fusion?

Chart chính: `14-fusion-ablation-recall20.png`.

Chart hỗ trợ: `01-allocation-modes-utility.png`,
`03-rescue-damage-vs-memory.png`.

### RQ2 — Việc gán đúng \(\beta_q\) cho đúng query có mang thông tin không, và gate nhận diện regime nào?

Chart kiểm định assignment: `15-beta-assignment-paired-effects.png`.

Chart cơ chế đã xác nhận: `12-beta-decile-mechanism-ndcg20.png`.

Chart giải thích allocator: `16-primary-gate-standardized-coefficients.png`.

Chart hỗ trợ: `04-dynamic-beta-distribution.png`,
`05-dynamic-beta-context-behavior.png`.

### RQ3 — Luật năm scalar có cạnh tranh với capacity lớn hơn không, và nhạy
đến đâu với RRF constant và fusion operator?

Chart chính: `11-allocation-capacity-controls.png` và
`06-rrf-k-sensitivity.png`.

Chart operator: `10-fusion-operator-rrf-vs-combsum.png`.

### RQ4 — Dynamic allocation thêm bao nhiêu inference cost?

Chart chính: `07-inference-performance.png`.

---

## 4. Kịch bản main talk 12–15 phút

## Slide 1 — Title và thesis

**Thời lượng:** 30–40 giây.

**Visual:** tiêu đề paper và một dòng công thức \(\beta_q\); chưa cần chart.

**Nội dung trên slide:**

- CEARF-N: Bounded Out-of-Fit Dynamic Rank Allocation.
- Training-only OOF allocation.
- Five learned scalars.
- Bounded per-query correction.

**Lời nói:**

> Session recommendation thường kết hợp memory và neural evidence bằng một
> trọng số cố định cho mọi query. Nhưng mức đáng tin của hai nguồn thay đổi
> theo context. CEARF-N học một luật phân bổ liên tục theo query từ out-of-fit
> training ranks. Validation không chọn beta; nó chỉ giữ vai trò audit hoặc
> upstream expert selection. Gate có năm scalar và mỗi query chỉ được hiệu
> chỉnh tối đa 0.10 quanh global prior.

**Chuyển slide:**

> Trước hết, vì sao một global coefficient là chưa đủ về mặt phương pháp?

---

## Slide 2 — Khoảng trống nghiên cứu

**Thời lượng:** 50–60 giây.

**Visual:** sơ đồ hai expert cùng đổ vào một nút fixed \(\beta\), sau đó đổi
thành \(\beta_q\).

**Ba ý trên slide:**

1. Memory mạnh khi local transition/session overlap rõ.
2. Neural có thể bổ sung semantic/generalization evidence.
3. Grid-search một \(\beta\) trên validation chỉ tạo một policy toàn cục.

**Lời nói:**

> Validation tuning một scalar là hợp lệ, nhưng nó không học được policy
> target-free theo từng query. Nếu fit allocator trực tiếp trên prediction của
> expert ở chính training source của expert, ta lại có self-fit artifact.
> Vì vậy bài toán không phải “thử thêm nhiều beta”, mà là học query-wise
> allocation từ prediction out-of-fit, với capacity đủ nhỏ để audit.

**Claim boundary:**

Không nói validation tuning là sai. Nói nó giải quyết một bài toán khác:
global model selection, không phải conditional allocation.

---

## Slide 3 — Phương pháp CEARF-N

**Thời lượng:** 75–90 giây.

**Visual:** `00-method-architecture.png`. Hình đã tách hai lane
training/inference và đặt hai phương trình ngay trong luồng.

**Visual backup khi reviewer hỏi “gate này có gì hơn router?”:**
`17-bounded-pair-certificate.png`.

**Lời nói:**

> Expert thứ nhất là CEARF memory, tổng hợp transition, neighbour-session và
> popularity evidence thành một memory ordering. Expert thứ hai là PASGR, một
> semantic GRU gọn, sinh neural ordering. Hai score gốc không cần cùng scale:
> mỗi ordering được chuyển sang reciprocal-rank evidence trong \([0,1]\).
>
> Trên OOF training queries, stage một học global prior
> \(\beta_{\mathrm{OOF}}\) liên tục. Stage hai đóng băng prior và chỉ học ba
> coefficient cùng một bias. Gate nhìn context length, last-item frequency và
> tail flag — tất cả có trước khi biết target.
>
> Với \(\delta_q=\beta_q-\beta_{\mathrm{OOF}}\) và
> \(D_q(x)=\rho_N(x)-\rho_M(x)\), mọi pair thay đổi đúng
> \(\delta_q[D_q(a)-D_q(b)]\). Vì vậy pair có global margin lớn hơn mức
> expert-disagreement tối đa được certificate là không thể đảo. Đây là
> controlled adaptation, không phải router tùy ý viết lại ranking.

**Điểm novelty phải nói rõ:**

> OOF stacking, conditional combination, RRF và GRU đều không mới riêng lẻ.
> Novelty hẹp nằm ở five-scalar allocator có rank-only expert interface và
> target-free inference features trên frozen heterogeneous experts, cùng
> bounded pair-intervention certificate và same-multiset assignment control.

---

## Slide 4 — Protocol không rò rỉ

**Thời lượng:** 60–75 giây.

**Visual:** `00-protocol-lineage.png`.

**Lời nói:**

> Trước khi fit, chúng tôi khai báo 5.000 validation query bằng stable hash.
> Từ training sources còn lại, tối đa 1.000 complete sessions dùng để khóa
> memory profile và 4.000 sessions khác dùng để học allocator. Các source này
> vắng hoàn toàn khỏi inner expert fit. Expert dự đoán chúng như dữ liệu
> out-of-fit. Sau đó split fingerprint, expert identity, feature statistics
> và gate state được serialize trước test.
>
> Điểm cần phân biệt là expert có thể được refit trên phần training được phép
> cho final test, nhưng allocator parameters đã đóng băng. Không có beta grid
> trên validation hay test.

**Claim boundary:**

Protocol này chứng minh provenance của allocation; nó không tự động bảo đảm
test improvement.

---

## Slide 5 — RQ1: Fusion có thực sự cần thiết?

**Chart:** `14-fusion-ablation-recall20.png`.

**Thời lượng:** 75–90 giây.

**Kết quả đã xác nhận từ artifact 3×3:**

| Domain | Best endpoint R@20 | Dynamic fused R@20 | Gain tuyệt đối | Gain tương đối |
|---|---:|---:|---:|---:|
| Video Games | 0.12571 | 0.14735 | +0.02164 | +17.2% |
| Baby Products | 0.04873 | 0.05669 | +0.00796 | +16.3% |
| Diginetica | 0.49428 | 0.51701 | +0.02273 | +4.6% |

**Lời nói:**

> Đây là result mạnh và đơn giản nhất. Trên cả ba domain, fused CEARF-N vượt
> endpoint mạnh hơn. Mức tăng R@20 là 17.2% trên Video, 16.3% trên Baby và
> 4.6% trên Diginetica. Điều này xác nhận memory và neural expert mang bằng
> chứng bổ sung, kể cả khi một endpoint riêng lẻ yếu.
>
> Tuy nhiên, đây chủ yếu là bằng chứng cho **complementarity của fusion**.
> Nó chưa chứng minh query-wise gate tốt hơn global allocation; câu hỏi đó
> được tách riêng ở các slide sau.

**Câu không nên nói:**

> “Dynamic gate tạo ra toàn bộ mức tăng 17.2%.”

**Câu nên nói:**

> “Dynamic CEARF-N giữ trọn lợi ích bổ sung của hai expert; phần đóng góp riêng
> của query-wise correction được đánh giá bằng global và permutation controls.”

---

## Slide 6 — RQ1: Tách fusion gain khỏi allocation gain

**Chart:** `01-allocation-modes-utility.png`.

**Thời lượng:** 60–75 giây.

**Lời nói:**

> Equal mixing đã tạo phần lớn bước nhảy từ endpoint lên fused system. OOF
> global cải thiện cách phân bổ chung, short/long là coarse conditional policy,
> và dynamic \(\beta_q\) là policy liên tục. So với OOF global, mean utility
> của dynamic thay đổi:
>
> - Video: \(+0.00021985\);
> - Baby: \(+0.00001548\);
> - Diginetica: \(+0.00001917\).
>
> Video dương ở cả ba seed và paired interval nằm trên 0 trong audit hiện tại.
> Baby và Diginetica có mean dương rất nhỏ nhưng interval cắt 0. Vì residual
> bị chặn trong 0.10, khoảng cách nhỏ là expected behavior chứ không nên được
> phóng đại.

**Expected interpretation:**

- Endpoint \(\rightarrow\) equal mixing: giá trị của evidence complementarity.
- Equal/global \(\rightarrow\) dynamic: giá trị biên của conditional
  assignment.
- Dynamic không “thắng lớn” global là phù hợp với thiết kế conservative.

**Chuyển slide:**

> Aggregate delta nhỏ đặt ra câu hỏi khó hơn: gate có thật sự gán đúng beta cho
> đúng query, hay chỉ tạo một distribution beta trông có vẻ động?

---

## Slide 7 — RQ2: Assignment có thông tin hay không?

**Chart ưu tiên:** `15-beta-assignment-paired-effects.png`.

**Trạng thái:** artifact 3 domain × 3 seed đã hoàn chỉnh.

**Thời lượng:** 70–90 giây.

**Control được kiểm tra:**

- Challenger: primary learned `dynamic_delta_010`.
- Baseline: deterministic permutation của **chính multiset
  \(\{\beta_q\}\)** đó.
- Hai hệ thống có cùng expert ranks và cùng beta distribution.
- Chỉ query-to-beta matching bị phá.
- So sánh paired cho nDCG@6/10/20 và Utility.

**Kết quả:**

- Video, nDCG@20 primary minus permuted:
  **+0.000146** \([+0.000082,+0.000210]\).
- Video, nDCG@6 và nDCG@10 cũng có interval trên 0.
- Diginetica, nDCG@10:
  **+0.000126** \([+0.000033,+0.000221]\); nDCG@20 sát biên nhưng cắt 0.
- Baby: không có contrast nào phân biệt được khỏi 0.

**Lời nói:**

> Đây là ablation trực tiếp nhất của mechanism. Chúng tôi giữ nguyên chính xác
> mọi giá trị beta nhưng hoán vị chúng giữa các query. Capacity, mean, variance
> và support của beta vì vậy không đổi. Trên Video, assignment thật tăng
> nDCG@20 0.000146 với toàn bộ interval trên 0; Diginetica cho tín hiệu hỗ trợ
> ở nDCG@10. Kết quả đến từ việc ghép context với allocation, không phải chỉ
> từ distribution beta. Baby là negative boundary: assignment chưa có signal
> phát hiện được trên domain đó.

---

## Slide 8 — RQ2: Gate nhận diện regime nào?

**Chart chính:** `12-beta-decile-mechanism-ndcg20.png`.

**Chart giải thích:** `16-primary-gate-standardized-coefficients.png`.

**Thời lượng:** 90 giây.

**Kết quả đã xác nhận cho Chart 12:**

- Video: neural-minus-memory nDCG@20 của D8–D10 cao hơn D1–D3 khoảng
  **+0.01614**, và dấu dương nhất quán qua ba seed.
- Baby và Diginetica không có ordered gap ổn định cùng chiều.

**Lời nói:**

> Aggregate gain không phải bằng chứng duy nhất của allocator. Ta sắp query
> theo learned beta rồi đo realized neural-minus-memory advantage. Trên Video,
> nhóm beta cao có neural advantage lớn hơn nhóm beta thấp 0.0161 nDCG@20,
> nhất quán qua ba seed. Gate vì vậy tạo một ordering có ý nghĩa của regime
> expert advantage trên domain này.
>
> Baby và Diginetica không cho cùng ordering ổn định. Kết luận đúng là
> **domain-conditional mechanism**, không phải universal routing law. Một
> controller bị chặn và co về global prior nên thay đổi ít khi OOF context
> không cung cấp signal ổn định.

**Cách đọc Chart 16 khi artifact đã có:**

- Dấu dương: feature tăng làm gate phân bổ nhiều neural mass hơn.
- Dấu âm: feature tăng làm gate phân bổ ít neural mass hơn.
- Whisker là seed SD, không phải confidence interval.
- Chỉ gọi một dấu là “stable tendency” nếu ba giá trị seed cùng dấu hoặc error
  bar không đi qua 0.
- Không diễn giải “tail gây ra neural utility”; coefficient chỉ mô tả policy
  được học trên standardized OOF features.

**Kết quả Chart 16:**

- Video: length \(-0.0766\), last-item frequency \(-0.0914\), tail
  \(+0.0185\); cả ba seed đều âm cho length/frequency và không âm cho tail.
- Diginetica: length \(-0.0553\) ổn định; frequency và tail thay dấu theo seed.
- Baby: tail coefficient âm ở cả ba seed; length và frequency đổi dấu, nên
  không có directional story rộng hơn.

**Câu chốt:** trên Video, gate chuyển neural mass về query ngắn hơn và
last-item ít phổ biến hơn; đây là mô tả policy, không phải causal effect.

---

## Slide 9 — RQ3: Robustness trong allocation family

**Charts:** `11-allocation-capacity-controls.png` và
`06-rrf-k-sensitivity.png`.

**Thời lượng:** 75–90 giây.

**Lời nói cho Chart 11:**

> Mỗi hàng thay đúng một trục: global, head/tail, short/long×head/tail,
> \(\Delta=.05/.10/.20\), và permuted assignment. Tất cả dùng cùng frozen OOF
> expert ranks. Vì vậy đây là comparison của allocation law, không phải một
> cuộc rerun expert không kiểm soát.

Trên Video, primary \(U=.11281\), coarse short/long×head/tail \(=.11278\),
permuted \(=.11271\), và \(\Delta=.20\) \(=.11292\). Sai khác đều cỡ
\(10^{-4}\). \(\Delta=.10\) là primary duy nhất được giữ cố định qua mọi
domain/seed; bảng sensitivity không được dùng để thay nó bằng
\(\Delta=.20\) sau test. Trên Baby/Diginetica, các capacity variants cũng
nằm trong cùng thang rất nhỏ. Bằng chứng hỗ trợ **bounded parsimony**, không
phải claim primary luôn là max.

**Kết quả RRF sensitivity đã xác nhận:**

| Domain | \(k=10\) | \(k=20\) | \(k=60\) |
|---|---:|---:|---:|
| Video | 0.11219 | **0.11281** | 0.11163 |
| Baby | 0.04261 | 0.04343 | **0.04354** |
| Diginetica | 0.41271 | **0.41332** | 0.41227 |

**Lời nói:**

> Primary \(k=20\) được giữ cố định qua mọi domain/seed, không được chọn lại
> từ bảng sensitivity. Nó tốt nhất trên Video và Diginetica; trên Baby,
> \(k=60\) chỉ cao hơn khoảng
> \(1.1\times10^{-4}\). Vì vậy conclusion không dựa trên một RRF constant
> mong manh.

**Claim boundary:**

Không gọi \(k=20\) là universally optimal. Chỉ nói kết luận ổn định trong ba
giá trị được kiểm tra và \(k=20\) không phải post-hoc selection.

---

## Slide 10 — RQ3: Fusion-operator sensitivity

**Chart:** `10-fusion-operator-rrf-vs-combsum.png`.

**Thời lượng:** 45–60 giây.

### Chart 10 — Fusion operator

**Numerical claim đã khóa:**

- Dưới frozen dynamic \(\beta_q\), weighted RRF có R@20 cao hơn normalized
  CombSUM lần lượt `0.00133`, `0.00134`, `0.00272` trên Video, Baby,
  Diginetica; cả ba interval đều không cắt 0.
- Trên nDCG@20, RRF và CombSUM gần như hòa ở Video, còn RRF cao hơn
  `0.00036` trên Baby và `0.00099` trên Diginetica.
- Ở allocation không học \(\beta=.5\), Diginetica đảo chiều: CombSUM cao hơn
  RRF `0.00194` R@20 và `0.00122` nDCG@20.

**Phạm vi control phải nói rõ:**

- Weighted RRF so với per-query, per-expert min–max CombSUM.
- Cùng frozen expert outputs và cùng union của persisted top-120 candidates.
- Đây không phải full-catalog score normalization.
- Dynamic \(\beta_q\) được giữ frozen; equal \(\beta=.5\) không có fitted
  parameter.

**Lời nói:**

> Khi giữ frozen dynamic allocation, weighted RRF có Recall@20 cao hơn
> normalized CombSUM trên cả ba miền. Nhưng equal mixing đảo chiều trên
> Diginetica. Vì vậy đây là bằng chứng cho cặp **allocator + rank operator**
> đã đề xuất, không phải tuyên bố RRF luôn tốt hơn mọi score-fusion.

---

## Slide 11 — External systems: CEARF-N đứng ở đâu?

**Chart:** `08-external-baseline-comparison-recall20.png`.

**Thời lượng:** 60–75 giây.

**Lời nói:**

> CEARF-N dẫn các comparator ID-only trên hai Amazon domain, nhưng không vượt
> NARM trên Diginetica. Đây là system comparison có disclosure: CEARF-N dùng
> cached text teacher — E5-small được document trên Video và TF–IDF/SVD trên
> Baby/Diginetica — trong khi các hàng external là ID-only.
>
> Vì vậy chart này trả lời “hệ thống hoàn chỉnh cạnh tranh đến đâu”, không trả
> lời “kiến trúc CEARF-N hơn NARM bao nhiêu khi cùng metadata”. Matched-teacher
> Amazon audit được dùng cho attribution riêng; Diginetica chưa có matched
> teacher comparator.

**Điểm nên chủ động nói:**

- Diginetica NARM R@20 khoảng 0.53406, cao hơn CEARF-N khoảng 0.51701.
- CEARF-N không được định vị là universal best model.
- Central causal contrast của paper là allocation trên **identical expert
  ranks**, không phải external rows.

---

## Slide 12 — RQ4: Inference cost

**Chart:** `07-inference-performance.png`.

**Thời lượng:** 50–60 giây.

**Số hiện có trong benchmark artifact:**

| Domain | Feature + gate | Post-processing gồm RRF | Throughput post-processing |
|---|---:|---:|---:|
| Video | 0.988 \(\mu s/q\) | 70.13 \(\mu s/q\) | 14.3k q/s |
| Baby | 0.859 \(\mu s/q\) | 71.05 \(\mu s/q\) | 14.1k q/s |
| Diginetica | 0.631 \(\mu s/q\) | 65.93 \(\mu s/q\) | 15.2k q/s |

**Lời nói:**

> Phần thực sự mới — ba feature và bounded linear gate — dưới một microsecond
> mỗi query. Khi cộng RRF trên hai top-120 lists, dynamic post-processing vẫn
> chỉ khoảng 65–71 microsecond, tương đương 14–15 nghìn query mỗi giây trong
> warm state. Expert retrieval và neural full-catalog scoring mới là phần chi
> phối latency.

**Claim boundary:**

- Panel post-processing là timing đo trực tiếp.
- Panel expert-inclusive là **cross-run composition estimate**, không phải một
  end-to-end run đồng hồ duy nhất.
- Không bao gồm loading, training, OOF calibration hoặc metric computation.
- Hardware: Apple M2 Pro, 32 GiB RAM.

---

## Slide 13 — Kết luận

**Thời lượng:** 40–50 giây.

**Ba takeaway trên slide:**

1. **Complementarity:** fused R@20 vượt best endpoint 4.6–17.2%.
2. **Mechanism:** bounded OOF gate nhận diện ordered neural-advantage regimes
   trên Video; assignment control quyết định mức claim rộng hơn.
3. **Deployability:** năm scalar, ba target-free inference features,
   allocation dưới \(1\,\mu s/q\).

**Lời kết:**

> CEARF-N không cố thay mọi expert bằng một kiến trúc lớn hơn. Nó làm cho việc
> kết hợp expert trở thành một đối tượng học được, có provenance và có giới
> hạn. Kết quả mạnh nhất là complementarity nhất quán trên ba domain; kết quả
> mới về query-wise allocation rõ nhất trên Video; và khi signal yếu, gate bị
> buộc ở gần global prior thay vì tự do overfit.

---

## 5. Backup slides

### Backup A — `13-dynamic-vs-global-six-metric-paired-delta.png`

**Dùng để trả lời:** “Kết quả có chỉ dựa vào Recall@20 không?”

Nói:

> Chúng tôi báo Recall và nDCG tại 6, 10 và 20. Chart giữ nguyên dấu và paired
> query-level interval cho cả sáu metric. Utility chỉ là trung bình
> \((R@6+R@20)/2\), không thay thế full metric reporting.

Không nói mọi metric đều dương. Đọc đúng từng point/CI trên chart cuối.

### Backup B — `03-rescue-damage-vs-memory.png`

**Dùng để trả lời:** “Fusion cải thiện bằng cách nào ở mức query?”

Nói:

> Rescue là memory miss trở thành fused hit; damage là memory hit bị mất.
> Net rescue cho thấy fusion thay đổi membership thực, không chỉ đổi thứ tự
> bên trong top-20.

Không dùng chart này để quy toàn bộ net rescue cho dynamic correction; chart
so với memory endpoint bao gồm cả fusion effect.

### Backup C — `04-dynamic-beta-distribution.png`

**Dùng để trả lời:** “Beta có thực sự thay đổi?”

Mean beta đã xác nhận:

- Video: 0.5716;
- Baby: 0.5411;
- Diginetica: 0.4204.

Phân phối khác hằng số, nhưng distribution alone không chứng minh assignment
hữu ích; Chart 15 mới là control đúng.

### Backup D — `05-dynamic-beta-context-behavior.png`

**Dùng để trả lời:** “Feature tác động thế nào?”

Chart mô tả mean-centered beta theo quartile context length/frequency và
head/tail. Đây là descriptive behavior. Chart 16 mới là coefficient trực tiếp
trên standardized OOF features.

### Backup E — provenance

Đưa các con số split fingerprint, source-disjoint OOF construction, manifest
freeze và exact replay audit. Mục đích là trả lời leakage/protocol, không phải
performance.

---

## 6. Q&A defenses

### Q1. “Validate beta là điều hiển nhiên; novelty ở đâu?”

**Trả lời:**

> Chúng tôi không chọn beta trên validation. Một grid trên validation sinh một
> scalar toàn cục. CEARF-N học liên tục một prior từ OOF train ranks, sau đó
> học một hàm theo query từ ba inference features không chứa target quanh prior
> đó. Điểm mới là conditional allocation với source-disjoint OOF supervision,
> rank-only expert interface và residual bound, không phải việc thêm một
> validation loop.

### Q2. “Method quá đơn giản, chỉ có linear gate?”

**Trả lời:**

> Đó là lựa chọn thiết kế có chủ đích. Allocator chỉ cần quyết định trust giữa
> hai expert, không cần học lại sequence representation. Năm scalar giúp audit,
> giảm selection capacity và cho formal bound trong fused-score space. Các
> linear/MLP capacity controls kiểm tra xem thêm capacity có thật sự cần hay
> không.

### Q3. “Nếu dynamic-over-global gain nhỏ, vì sao cần dynamic?”

**Trả lời:**

> Có hai cấp kết quả. Fusion complementarity tạo gain lớn so với endpoint.
> Dynamic gate là bounded refinement tối đa 0.10, nên expected aggregate delta
> nhỏ. Giá trị khoa học của allocator được kiểm tra thêm bằng assignment
> permutation và expert-advantage ordering, thay vì chỉ dựa vào mean metric.

### Q4. “Tại sao không học beta trực tiếp từ expert scores?”

**Trả lời:**

> Primary gate chỉ dùng feature có trước scoring để tránh gate biến thành một
> target-proxy hoặc một high-capacity score calibrator. Rank overlap và các
> expert diagnostics được dành cho capacity ablation, không được post-hoc thêm
> vào primary.

### Q5. “RRF có phải contribution mới không?”

**Trả lời:**

> Không. RRF là fusion primitive đã biết. Contribution là cách học bounded
> query allocation trên reciprocal-rank evidence. \(k\)-sensitivity và
> CombSUM control tách primitive khỏi allocation claim.

### Q6. “Tại sao \(k=20\), trong IR thường dùng \(k=60\)?”

**Trả lời:**

> \(k=20\) được predeclare và không được chọn lại trên test. Sensitivity cho
> 10/20/60 cho thấy 20 tốt nhất trên Video và Diginetica; trên Baby, 60 chỉ hơn
> khoảng \(1.1\times10^{-4}\). Do đó conclusion không phụ thuộc vào chọn một
> \(k\) duy nhất.

### Q7. “Permutation beta kiểm tra chính xác điều gì?”

**Trả lời:**

> Nó giữ nguyên multiset beta, do đó giữ mean, variance, range và allocation
> capacity; chỉ phá mapping giữa query và beta. Paired difference vì vậy cô
> lập giá trị của assignment. Chúng tôi chỉ claim positive assignment ở
> domain/metric có interval hỗ trợ.

### Q8. “Coefficient dương có nghĩa tail item được neural xử lý tốt hơn?”

**Trả lời:**

> Không nhất thiết. Coefficient cho biết policy phân bổ nhiều neural mass hơn
> khi standardized feature tăng, conditional on các feature còn lại. Nó là
> association của fitted allocator, không phải causal effect hoặc proof về
> neural accuracy.

### Q9. “Metadata comparison có công bằng không?”

**Trả lời:**

> External chart là system-level comparison và được ghi rõ comparator ID-only.
> Attribution chính của dynamic gate dùng cùng frozen expert ranks nên không
> có metadata asymmetry giữa challenger và baseline. Matched-teacher NARM audit
> được báo riêng trên hai Amazon domain; chưa có matched-teacher Diginetica.

### Q10. “Vì sao Diginetica thua NARM?”

**Trả lời:**

> Primary PASGR-based CEARF-N thua ID-only NARM trên Diginetica, nên đây là
> giới hạn của neural expert primary. Expert-swap control thay PASGR bằng
> ID-only NARM trong cùng allocator và đạt R@20 .53939, cao hơn external NARM
> .53406. Vì vậy vấn đề là chọn expert cho domain, không phải allocator không
> dùng được khi expert mạnh hơn.

### Q11. “Validation và test khác distribution thì sao?”

**Trả lời:**

> Không có model-selection procedure nào tự bảo đảm transfer dưới distribution
> shift. Ở đây allocator thậm chí không học từ validation labels; nó học từ
> training-only OOF queries và chỉ được đánh giá trên test. Bài không biến
> validation-to-test transfer thành novelty claim. Domain/seed/full-metric
> reporting cho thấy phạm vi generalization thực nghiệm.

### Q12. “Chỉ ba dataset có đủ không?”

**Trả lời:**

> Ba domain bao gồm hai sparse Amazon regimes và một session benchmark lớn,
> đều full-catalog và ba matched seeds. Chúng tôi không thêm dataset không có
> valid-query protocol chỉ để tăng số lượng. Giới hạn ba domain được ghi rõ;
> claim cũng được giới hạn theo domain, đặc biệt mechanism claim tập trung ở
> Video.

### Q13. “Bootstrap unit trên Diginetica là gì?”

**Trả lời:**

> Artifact không giữ original session ID cho mọi test record, nên paired
> bootstrap dùng smallest recoverable query identifier. Chúng tôi gọi đúng là
> query-level bootstrap, không gọi session-clustered.

### Q14. “Inference timing có bao gồm neural model không?”

**Trả lời:**

> Panel phải đo trực tiếp là dynamic post-processing sau khi hai expert lists
> đã có. Expert-inclusive panel là cross-run estimate được label rõ, không phải
> co-timed end-to-end measurement. Bản thân feature+gate dưới một microsecond
> mỗi query.

### Q15. “Positive result có phải cherry-pick Video không?”

**Trả lời:**

> RQ hỏi trên domain nào allocator nhận diện ordered regime; paper báo cả ba
> domain và nói thẳng Baby/Diginetica không ổn định. Video là answer của RQ,
> không phải dataset duy nhất được giữ lại. Endpoint complementarity vẫn dương
> trên cả ba domain.

---

## 7. Checklist ngay trước khi trình bày

1. Chạy lại chart generator sau khi final summary/control artifacts hoàn tất.
2. Kiểm tra `chart_manifest.json` có seeds 42/123/456 cho từng conditional
   chart.
3. Không dùng contact sheet cũ chỉ có chart 01–08.
4. Đối chiếu mọi số hard-code trong slide với source SHA của chart.
5. Với Chart 15, nói rõ primary-versus-permuted dùng cùng beta multiset.
6. Với Chart 16, ghi “mean ± seed SD” và “not causal”.
7. Với Chart 10, ghi rõ score normalization trên union top-120.
8. Với Chart 07, tách measured post-processing khỏi cross-run estimate.
9. Nếu một control còn pending, bỏ chart khỏi main deck; không thay bằng số
    ước lượng hoặc text mang hàm ý kết quả đã có.

## 8. Câu kết ngắn cho Q&A

> CEARF-N mạnh không phải vì gate lớn, mà vì gate nhỏ nhưng được học đúng chỗ:
> trên prediction out-of-fit, theo từng query, với một biên thích nghi có thể
> kiểm toán. Kết quả cho thấy fusion complementarity nhất quán; dynamic
> mechanism rõ nhất trên Video; và mọi claim rộng hơn đều được khóa bằng
> same-multiset reassignment, operator và runtime controls.
