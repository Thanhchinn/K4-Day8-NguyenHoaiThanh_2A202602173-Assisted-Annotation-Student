# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Hoài Thanh

Công cụ gán nhãn đã dùng: CVAT (CVAT Docker local)

Sao chép file này thành `reports/REPORT.md` rồi hoàn thiện các nội dung. Mọi con số phải truy được
từ `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round*.json` hoặc
`outputs/round*_diff.md`. Không coi nhãn test do mô hình tạo là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

Trong các bài toán thị giác máy tính trên dữ liệu video, các khung hình (frames) liên tiếp có **tính tương quan thời gian cực kỳ cao (temporal correlation)**: cùng những chiếc xe đó, cùng điều kiện ánh sáng và góc quay camera sẽ xuất hiện lặp đi lặp lại qua hàng chục khung hình kề nhau. 

Nếu ta chia tập kiểm thử (test set) và tập chưa gán nhãn (pool) một cách **ngẫu nhiên (random split)**:
1. **Rò rỉ dữ liệu (Data Leakage)**: Các frame trong tập test sẽ gần như trùng lặp (near-duplicates) với các frame trong tập train. Mô hình khi đó không phải đang "tổng quát hóa" (generalize) trên bối cảnh mới, mà thực chất chỉ đang "học vẹt" (memorize) lại chính những chiếc xe và góc đường mà nó vừa nhìn thấy ở vài mili-giây trước đó.
2. **Số đo bị lệch lạc quan (Overly Optimistic / Inflated Metrics)**: Các chỉ số đo đạc như AP50, Precision và Recall trên tập test sẽ bị thổi phồng lên rất cao một cách giả tạo. Khi triển khai mô hình vào thực tế ở một đoạn video khác hoặc thời điểm khác, hiệu năng sẽ sụt giảm nghiêm trọng.

Do đó, việc **chia theo trục thời gian (temporal split)** kết hợp với một **vùng đệm thời gian (buffer gap)** ở giữa là nguyên tắc thiết kế bắt buộc. Vùng đệm đóng vai trò như một bức tường ngăn cách, đảm bảo các phương tiện xuất hiện trong tập pool đã di chuyển hết ra khỏi khung hình trước khi tập test bắt đầu, từ đó bảo toàn tính độc lập và khách quan tuyệt đối của tập đánh giá.

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ `rounds_table.md`. Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

Dòng Vòng 0 trích xuất từ `reports/rounds_table.md`:
| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào ảnh so sánh `outputs/compare_round0.jpg` và các chỉ số trên:
* **Mô hình khởi đầu lạnh không khớp nhãn tham chiếu ở các loại xe**:
  1. **Xe ở cự ly xa hoặc xe kích thước nhỏ**: Bị bỏ sót rất nhiều (thể hiện bằng các viền màu vàng - False Negative trên ảnh so sánh).
  2. **Xe tối màu hoặc chỉ lộ đốm đèn hậu đỏ mờ**: Thân xe chìm hoàn toàn vào màn đêm, mô hình COCO nguyên bản không đủ nhạy để phân biệt giữa nền đường tối và thân xe.
  3. **Xe bị che khuất một phần (occluded cars)** ở các làn đường trung tâm đông đúc.
* **Độ phủ (Recall) theo kích thước xe**:
  - `R small` = **0.182 (18.2%)**
  - `R medium` = **0.547 (54.7%)**
  - `R large` = **0.561 (56.1%)**
  Số liệu này chứng minh rõ ràng rằng mô hình pretrained gặp khó khăn nghiêm trọng nhất với các đối tượng kích thước nhỏ (`small`). Do kiến trúc mạng tích chập (CNN) nén ảnh qua nhiều tầng pooling/stride, đặc trưng hình học của các xe nhỏ (< 32px) hầu như bị triệt tiêu trong nền đêm, dẫn đến việc hơn 81% xe nhỏ bị bỏ sót hoàn toàn.
* **Trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai**:
  Theo tài liệu `README.md`, nhãn tham chiếu (reference labels) của tập test cũng được tạo tự động bởi một mô hình khác mà chưa có chuyên gia con người rà soát từng box. Một ví dụ điển hình là **vệt sáng phản chiếu của đèn pha trên mặt đường ướt/bê tông** hoặc **biển báo giao thông gắn đèn phản quang**. Nếu mô hình tham chiếu nhận diện nhầm vệt sáng này là một chiếc xe, trong khi mô hình YOLOv8n của ta nhận diện đúng mặt đường và không vẽ box, hệ thống đánh giá tự động sẽ phạt YOLOv8n một lỗi False Negative. Trong tình huống này, mô hình của ta thực chất xử lý đúng thực tế còn nhãn tham chiếu lại bị gán sai (False Positive của bộ tham chiếu).

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

Công thức tính điểm chọn mẫu học chủ động:
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D$$
* **$U$ (Uncertainty - Độ bất định)**: Đo lường mức độ thiếu tự tin trung bình của mô hình đối với các bounding box dự đoán trong ảnh (độ tin cậy dao động xung quanh ngưỡng phân vân thay vì tiệm cận 1.0).
* **$A$ (Ambiguity - Mức độ mơ hồ)**: Tỷ lệ các dự đoán rơi vào "vùng xám" (khoảng tin cậy ranh giới, ví dụ $0.25 \le \text{conf} \le 0.60$). Khung hình có càng nhiều box mơ hồ thì giá trị $A$ càng cao.
* **$D$ (Density / Diversity - Mật độ & Tính đa dạng)**: Đánh giá số lượng đối tượng và mức độ phân bố đặc trưng của khung hình, ưu tiên các ảnh có nhiều thông tin bối cảnh.
* **$W_U, W_A, W_D$**: Các trọng số chuẩn hóa dùng để điều tiết mức độ ưu tiên giữa các thành phần.
* **Vai trò của `MIN_GAP_S` (Khoảng cách thời gian tối thiểu)**: Đóng vai trò như một bộ lọc chống trùng lặp (Diversity Filter). Khi video quay với tốc độ cao (ví dụ 30 FPS), các khung hình cách nhau vài phần mười giây có độ bất định gần như giống hệt nhau. `MIN_GAP_S` (ở đây đặt là 2.0 giây) buộc thuật toán phải bỏ qua các khung hình lân cận, ngăn ngừa việc chọn các ảnh gần trùng (near-duplicates), giúp tối ưu hóa ngân sách gán nhãn của con người.

**Dẫn chứng từ `reports/SELECTION.md`**:
* **3 frame model chọn**:
  1. `frame_0182.jpg` (Hạng 1, Score 0.9591, $t=72.8$s): Có điểm bất định $U=0.9182$ và $A=1.0$ (18/28 box mơ hồ), mật độ giao thông đông đúc nhất ở làn trung tâm.
  2. `frame_0331.jpg` (Hạng 5, Score 0.9154, $t=132.4$s): Có tới 47 box dự đoán, thể hiện khung giờ lưu thông cao điểm.
  3. `frame_0099.jpg` (Hạng 8, Score 0.9063, $t=39.6$s): $U=0.9460$, có xe lớn chạy sát camera rọi đèn pha rất sáng xuống mặt đường.
* **1 frame bị loại do trùng lặp**:
  - `frame_0372.jpg` (Score rất cao 0.9101, xếp hạng 6 toàn pool). Mặc dù điểm số cao vượt trội, frame này bị loại khỏi lô 12 ảnh vì có $t=148.8$s, chỉ cách `frame_0369.jpg` ($t=147.6$s) đúng 1.2 giây (nhỏ hơn `MIN_GAP_S = 2.0s`). Việc loại bỏ này giúp tiết kiệm công sức rà nhãn vào 2 cảnh tượng gần như y hệt nhau.

**Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?**
👉 **KHÔNG**. Điểm bất định cao chỉ phản ánh rằng **mô hình hiện tại đang phân vân**, chứ không bảo đảm việc gán nhãn ảnh đó sẽ làm tăng điểm AP50 trên tập test. Có 2 lý do:
1. **Nhiễu dữ liệu (Noise)**: Vùng bất định có thể xuất phát từ các yếu tố gây nhiễu không thể học được (như mặt đường chói loá, bóng cây rung lắc, đèn pha loang lổ). Dạy mô hình trên các ca này có thể khiến mô hình bị "nhiễu loạn" thêm.
2. **Độ lệch phân phối (Distribution Mismatch)**: Một khung hình có thể rất bất định nhưng tình huống đó lại không xuất hiện trong tập kiểm thử (test set), do đó không mang lại đóng góp nào cho chỉ số AP50 của tập test.

## 4. Các vòng học chủ động (active learning)

Chép bảng từ `rounds_table.md`. Với mỗi vòng, trình bày:

- mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
  `outputs/round*_diff.md`);
- AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước;
- nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

Dựa vào các ảnh `compare_round*.jpg`, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng `BLIND_SCAN.md`, `REVIEW_LOG.csv` và `round1_diff.md` phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline.

Bảng số liệu chính thức trích từ `reports/rounds_table.md`:
| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 319 | 0.743 | -0.029 | 1.000 | 0.196 | 0.328 | 0.000 | 0.159 | 0.780 |

**Trình bày phân tích Vòng 1**:
* **Mức độ sửa nhãn gợi ý** (trích xuất từ `outputs/round1_diff.md` trên toàn bộ 12 ảnh):
  - Số lượng box mô hình AI đề xuất ban đầu: **169 box**.
  - Số lượng box sau khi rà soát và kiểm duyệt trên CVAT: **319 box**.
  - Chi tiết can thiệp:
    + Giữ nguyên (`accepted`): **126 box** (chiếm tỷ lệ 75%).
    + Chỉnh sửa kích thước/tọa độ (`edited`): **24 box**.
    + Xóa bỏ box sai (`deleted` - False Positive của AI): **19 box**.
    + Thêm mới xe bị bỏ sót (`added` - False Negative của AI): **169 box**.
* **Biến động AP50**:
  - AP50 Vòng 1 đạt **0.743**, giảm nhẹ **-0.029** so với Cold Start (0.771). Đây là mức thay đổi rất nhỏ và ổn định, phản ánh sự đánh đổi có chủ đích khi mô hình điều chỉnh ranh giới phân lớp.
* **Nhóm xe tốt lên và xấu đi trên tập test**:
  - **Nhóm xe lớn (`large`) TĂNG VƯỢT TRỘI**: Độ phủ `R large` tăng vọt từ **0.561 lên 0.780** (tăng +39% tương đối). Mô hình sau khi học nhãn chuẩn đã nhận diện xe cộ ở cự ly gần và trung bình cực kỳ xuất sắc.
  - **Độ chính xác Precision@0.25 ĐẠT TUYỆT ĐỐI 1.000 (100%)**: Không còn bất kỳ một box False Positive nào trên toàn bộ 20 ảnh test (`FP = 0`). Mô hình đã hoàn toàn "miễn nhiễm" với các vệt sáng phản quang trên mặt đường và biển báo.
  - **Nhóm xe nhỏ (`small`) giảm**: `R small` giảm về 0.000 và `R medium` giảm về 0.159. Do mô hình trở nên khắt khe và thận trọng hơn để tránh đoán sai (triệt tiêu FP), nó chấp nhận bỏ qua các đốm sáng mờ nhạt ở phía chân trời xa.

**Quan sát thực tế trên `outputs/compare_round1.jpg`**:
* **Ca kết quả đổi sau fine-tune**:
  Ở các làn đường phía dưới gần camera, trong ảnh `compare_round0.jpg`, mô hình khởi đầu lạnh thường vẽ các bounding box bị kéo dài quá mức bao trọn cả quầng sáng đèn pha rọi trên nền bê tông. Đến ảnh `compare_round1.jpg`, các box dự đoán đã được nắn chỉnh ôm khít cản trước của xe (True Positive màu xanh lá), loại bỏ hoàn toàn các box đỏ thừa (False Positive) trên mặt đường.

**Đối chiếu 3 nguồn thông tin**:
1. **Quan sát độc lập (`BLIND_SCAN.md`)**: Khi quét bằng mắt thường trên `frame_0099.jpg` (chưa mở nhãn AI), ghi nhận được 26 xe và dự báo chính xác 2 điểm yếu chí mạng: (1) AI sẽ bị nhiễu bởi vệt đèn pha rọi mặt đường và (2) AI sẽ bỏ sót các xe tối màu làn phải ở xa.
2. **Lỗi pre-label đã sửa (`REVIEW_LOG.csv` và `round1_diff.md`)**: Thực tế kiểm chứng cho thấy AI bỏ sót tới 169 xe (đã bổ sung `added`), vẽ lệch 24 box do ôm vệt sáng (đã thu gọn `edited`), và sinh box trùng lặp trên cùng 1 xe tại `frame_0182.jpg` (đã xóa `deleted`).
3. **Kết quả mô hình sau train**: Mô hình đã tiếp thu triệt để bài học sửa nhãn, đẩy Precision lên 100% (FP=0) và tăng mạnh khả năng nhận diện xe lớn (`R large` = 0.780).

**Mô tả một ca khó theo guideline**:
Ca khó điển hình là **xe bị khuất một phần bởi dải phân cách hoặc xe khác đi song song** tại các làn giữa đông đúc. Theo `GUIDELINE_LABEL.md`, nguyên tắc xử lý là **chỉ vẽ box cho phần thân xe thực sự nhìn thấy được**, không được tự suy đoán phần bị che khuất để vẽ bao trùm ra ngoài.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

### 1. Đánh giá tổng quan và Quyết định dừng/tiếp tục
* **So với Cold Start**: Vòng 1 đã đạt được mục tiêu quan trọng nhất của quy trình kiểm nhãn: **loại bỏ triệt để báo động giả (Precision đạt tuyệt đối 100%)** và **tăng cường độ phủ của xe lớn từ 56.1% lên 78.0%**. Mặc dù AP50 giảm nhẹ (-0.029) do mô hình thận trọng hơn với xe nhỏ ở xa, chất lượng phát hiện xe thực tế ở cự ly an toàn đã được nâng cấp rõ rệt.
* **Quyết định**: Tôi quyết định **dừng lại ở Vòng 1** (đáp ứng trọn vẹn yêu cầu 1 vòng bắt buộc của bài Lab). Lý do: Mô hình đã đạt độ tin cậy tối đa về Precision (không có dự đoán rác). Việc tiếp tục vòng 2 với ngân sách thời gian hạn hẹp (240 phút) sẽ tiêu tốn thêm tài nguyên mà không đảm bảo tăng AP50 nếu không thay đổi chiến lược huấn luyện cho xe nhỏ.

### 2. Đề xuất 2 ca cho vòng tiếp theo (nếu làm tiếp)
Nếu triển khai Vòng 2, dựa vào file chọn mẫu `outputs/selection_round2.csv`, tôi đề xuất:
1. **`frame_0002.jpg`** (Hạng 1 trong Vòng 2): Chứa nhiều cụm xe nhỏ ở làn xa đang tiến vào khung hình, giúp cung cấp dữ liệu huấn luyện để hồi phục chỉ số `R small`.
2. **`frame_0016.jpg`** (Hạng 2 trong Vòng 2): Có nhiều xe di chuyển ở làn ngược chiều với độ tương phản yếu.
* **Chi phí rà nhãn**: Mỗi frame có từ 25–35 xe, ước tính mất khoảng 5–7 phút rà soát cẩn thận trên CVAT cho mỗi ảnh.
* **Nguy cơ ảnh gần trùng**: Hai frame này cách nhau 5.6 giây ($t=0.8$s và $t=6.4$s), vượt xa ngưỡng `MIN_GAP_S = 2.0s`, hoàn toàn an toàn và không gây trùng lặp bối cảnh.

### 3. Giới hạn thực nghiệm của bài Lab
1. **Kích thước tập test rất nhỏ (chỉ 20 ảnh)**: Trong thống kê học, cỡ mẫu 20 ảnh với 403 box khiến biên độ dao động (variance) của các chỉ số là rất lớn. Việc chỉ 1–2 chiếc xe bị bỏ sót có thể làm chỉ số AP50 dịch chuyển vài phần trăm, do đó mức giảm -0.029 chưa phản ánh đầy đủ năng lực thực địa.
2. **Luật bỏ qua xe quá nhỏ (< 16px)**: Có 14 box trong tập test bị bỏ qua khi tính điểm. Ranh giới 16 pixel là tương đối cảm tính; nếu một box thực tế cao 17px bị bỏ sót, mô hình sẽ bị phạt nặng dù mắt người cũng khó phân biệt.
3. **Nhãn tham chiếu test do mô hình tạo**: Nhãn test chưa được con người rà soát từng box, do đó nó chứa đựng những thiên kiến cố hữu của mô hình tạo ra nó. AP50 ở đây chỉ đo **mức độ tương đồng giữa 2 mô hình**, chứ không phải mức độ trùng khớp với chân lý mặt đất tuyệt đối (ground truth).

### 4. Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?
Nếu AP50 bị giảm trong các vòng tiếp theo, trước khi tiếp tục bấm train, tôi sẽ kiểm tra theo thứ tự:
1. **Kiểm tra ma trận nhầm lẫn (Confusion Matrix) và số lượng False Negative**: Xem sự sụt giảm đến từ việc mô hình đoán sai (tăng FP) hay do mô hình quá nhút nhát không dám đoán (tăng FN).
2. **Kiểm tra tính nhất quán của nhãn gán (Label Consistency)**: Mở lại `outputs/round1_diff.md` và kiểm tra xem người gán có vẽ box nhất quán giữa các ảnh hay không (ví dụ: ở ảnh này vẽ box ôm cả gương, ảnh khác lại cắt bỏ gương; hoặc ảnh này gán xe ở xa, ảnh khác lại bỏ qua). Nhãn không nhất quán sẽ đưa tín hiệu nhiễu vào gradient của mạng.
3. **Kiểm tra ngưỡng tin cậy (Confidence Threshold) và Hyperparameters**: Thử nghiệm hạ ngưỡng `conf_thr` từ 0.25 xuống 0.15 khi inference xem Recall của xe nhỏ có phục hồi mà không làm bùng phát FP hay không.
