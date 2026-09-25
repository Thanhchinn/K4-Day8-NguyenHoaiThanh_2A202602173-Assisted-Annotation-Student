# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box:

Nếu chỉ có ngân sách rà đúng 5 ảnh, tôi đề xuất top 5 frame ưu tiên sau:
1. **`frame_0182.jpg`** (Thứ tự: 1, Điểm score: 0.9591, Thời điểm: 72.8s, U: 0.9182, A: 1.0, D: 1.0): Đứng đầu toàn bộ pool 268 ảnh, có tới 18/28 box rơi vào vùng bất định (A=1.0). Đây là khung hình có mật độ xe cao, nhiều xe chen chúc ở làn giữa, đại diện tiêu biểu cho tình huống giao thông phức tạp.
2. **`frame_0369.jpg`** (Thứ tự: 2, Điểm score: 0.9324, Thời điểm: 147.6s, U: 0.9315, A: 0.8889, D: 1.0): Thời điểm ở nửa sau video (cách xa frame 0182 gần 75 giây), mật độ xe dày đặc (43 box dự đoán, 16 box bất định), mang lại bối cảnh giao thông độc lập cao.
3. **`frame_0326.jpg`** (Thứ tự: 4, Điểm score: 0.9155, Thời điểm: 130.4s, U: 0.9310, A: 0.8333, D: 1.0): Điểm bất định cao, thời điểm t=130.4s cách xa frame 0369 (>17 giây), xuất hiện nhiều xe di chuyển ở cả làn trong và làn ngoài.
4. **`frame_0099.jpg`** (Thứ tự: 8, Điểm score: 0.9063, Thời điểm: 39.6s, U: 0.9460, A: 0.7778, D: 1.0): Đại diện cho đoạn đầu video, điểm bất định U rất cao (0.9460). Frame này có xe lớn di chuyển gần camera rọi đèn pha chói xuống mặt đường bê tông, là ca kinh điển cần người phân định rõ giữa cản trước của xe và vệt sáng phản chiếu.
5. **`frame_0270.jpg`** (Thứ tự: 13, Điểm score: 0.8878, Thời điểm: 108.0s, U: 0.9089, A: 0.7778, D: 1.0): Đại diện cho đoạn giữa video (t=108.0s), giúp trải đều mẫu theo trục thời gian, tránh dồn cục bộ vào một khoảng thời gian ngắn.

*Quyết định xét ảnh gần trùng (Near-duplicates):* Tôi chủ động **loại bỏ `frame_0372.jpg`** (Thứ tự: 6, Score: 0.9101, Thời điểm: 148.8s) dù điểm số nằm trong Top 6. Lý do là frame này chỉ cách `frame_0369.jpg` (t=147.6s) đúng 1.2 giây (nhỏ hơn `MIN_GAP_S = 2.0s`). Các xe và bối cảnh ở hai frame này gần như trùng lặp hoàn toàn; việc gán cả hai sẽ lãng phí 20% ngân sách rà nhãn mà không bổ sung thêm thông tin phân phối mới cho mô hình.

---

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:
1. **`frame_0182.jpg`**: Trong CSV xếp hạng 1 với điểm số cao nhất 0.9591, A=1.0 (18 box bất định). Trên ảnh contact sheet (`outputs/selection_round1.jpg`), frame này thể hiện cụm xe dày đặc ở làn trung tâm, nhiều đốm đèn hậu đỏ và xe bị che khuất một phần.
2. **`frame_0331.jpg`**: Trong CSV xếp hạng 5 với score 0.9154, n_boxes = 47 (số lượng box dự đoán nhiều nhất trong cả lô 12 ảnh). Trên contact sheet, ảnh này thể hiện mật độ lưu thông cao nhất, các xe bám đuôi nhau san sát.
3. **`frame_0099.jpg`**: Trong CSV xếp hạng 8 với score 0.9063, U=0.9460. Trên contact sheet, ảnh thể hiện rõ vệt sáng đèn pha phản quang rất mạnh trên nền đường phía trước 2 xe gần camera, là nguyên nhân khiến AI phân vân giữa vệt sáng và thân xe.

---

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
* **Frame có điểm cao nhưng không chọn**: `frame_0372.jpg` (xếp thứ 6 trong CSV, score 0.9101, t=148.8s) và `frame_0368.jpg` (xếp thứ 9, score 0.9003, t=147.2s). Mặc dù điểm số thuộc top đầu, mô hình vẫn loại bỏ không chọn vào lô 12 ảnh vì vi phạm ràng buộc khoảng cách thời gian tối thiểu `MIN_GAP_S = 2.0s` so với `frame_0369.jpg` (t=147.6s). Đây là cơ chế lọc đa dạng (diversity filtering) cực kỳ cần thiết để chống thiên kiến do video quay liên tục với FPS cao.
* **Frame có điểm thấp vẫn nên xem**: `frame_0195.jpg` (xếp cuối cùng thứ 268 trong CSV, score 0.5721, U=0.6442, n_boxes=20). Điểm số thấp vì AI rất "tự tin" với các box nó tìm thấy. Tuy nhiên, AI có thể rơi vào bẫy "tự tin thái quá" (overconfidence) hoặc "mù hoàn toàn" (blind spot) với những xe màu đen tắt đèn chạy ở làn khuất mà AI hoàn toàn không sinh ra box nào. Do đó, kiểm tra ngẫu nhiên các frame điểm thấp giúp phát hiện các ca False Negative nghiêm trọng mà điểm bất định bỏ sót.

---

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
Phép chọn theo độ bất định (Uncertainty Sampling) chỉ đo lường mức độ phân vân của mô hình hiện tại trên các dự đoán mà nó tự tạo ra. Điều này **chưa chứng minh được**:
1. **Không bảo đảm AP50 sẽ tăng**: Việc đưa các ảnh bất định vào huấn luyện có thể giúp mô hình học thêm trường hợp khó, nhưng nếu dữ liệu có nhiều nhiễu (vệt đèn chói, xe quá mờ) hoặc nhãn gán không hoàn hảo, mô hình có thể bị phân tâm và giảm độ phủ trên các xe thông thường.
2. **Không phát hiện được điểm mù tuyệt đối (Unknown Unknowns)**: Nếu một chiếc xe nằm trong bóng tối mà mô hình bỏ sót 100% (không có bất kỳ box dự đoán nào ở ngưỡng > 0.1), thuật toán không ghi nhận độ bất định nào tại vị trí đó, dẫn đến việc frame chứa xe bị bỏ sót vẫn bị chấm điểm thấp và không được chọn vào vòng học chủ động.
