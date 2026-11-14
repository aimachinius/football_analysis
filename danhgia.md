Hiệu năng theo dõi đa đối tượng được đánh giá theo chuẩn MOTChallenge, với quy trình so khớp như sau:

Dữ liệu chuẩn (GT): các chuỗi khung hình thử nghiệm được gán nhãn thủ công và kiểm định độc lập. Mỗi cầu thủ, thủ môn và trọng tài có một ID duy nhất, bất biến trong suốt thời gian xuất hiện.
Ngưỡng ghép hình học: hộp bao dự đoán $B_{\text{pred}}$ được ghép với hộp chuẩn $B_{\text{gt}}$ nếu:
$$\text{IoU}(B_{\text{pred}}, B_{\text{gt}}) = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})} \ge 0.50$$
Ghép cặp Hungarian toàn cục: tại mỗi khung hình, thuật toán Hungarian tìm ánh xạ 1-1 tối ưu trên ma trận chi phí $\mathbf{C}_{i,j} = 1 - \text{IoU}$ (đặt $\infty$ nếu $\text{IoU} < 0.50$), từ đó xác định TP (cặp ghép hợp lệ), FP (hộp dự đoán không ghép được, tức phát hiện giả) và FN (hộp chuẩn bị bỏ sót).

Trên cơ sở đó, hệ thống được đánh giá bằng các độ đo sau:

IDF1 (Ristani et al., ECCV 2016): đo khả năng gán đúng một ID duy nhất cho mỗi đối tượng trong toàn bộ video. Đây là thước đo then chốt trong phân tích thể thao vì nhảy ID làm sai lệch quãng đường chạy và biểu đồ nhiệt.
$$\text{IDF1} = \frac{2 \cdot \text{IDTP}}{2 \cdot \text{IDTP} + \text{IDFP} + \text{IDFN}} = \frac{2 \cdot \text{IDP} \cdot \text{IDR}}{\text{IDP} + \text{IDR}}$$
với $\text{IDP} = \frac{\text{IDTP}}{\text{IDTP} + \text{IDFP}}$ và $\text{IDR} = \frac{\text{IDTP}}{\text{IDTP} + \text{IDFN}}$. Trong đó $\text{IDTP}$ là số khung hình đối tượng được phát hiện đúng vị trí và đúng ID; $\text{IDFP}$ là số khung hình phát hiện dư hoặc gán sai ID; $\text{IDFN}$ là số khung hình bỏ sót.
HOTA (Luiten et al., CVPR 2020): cân bằng chất lượng phát hiện ($\text{DetA}$) và chất lượng liên kết danh tính ($\text{AssA}$), khắc phục việc $\text{MOTA}$ thiên về phát hiện.
$$\text{DetA} = \frac{|TP|}{|TP| + |FP| + |FN|}, \qquad \text{AssA} = \frac{1}{|TP|} \sum_{c \in TP} \frac{|\text{TPA}(c)|}{|\text{TPA}(c)| + |\text{FPA}(c)| + |\text{FNA}(c)|}$$
$$\text{HOTA} = \sqrt{\text{DetA} \cdot \text{AssA}}$$
MOTA (Bernardin & Stiefelhagen, 2008): tổng hợp bỏ sót ($FN$), báo động giả ($FP$) và hoán đổi danh tính ($IDSW$), với $GT_t$ là số đối tượng thực tế tại khung hình $t$.
$$\text{MOTA} = 1 - \frac{\sum_{t=1}^T \left( FN_t + FP_t + IDSW_t \right)}{\sum_{t=1}^T GT_t}$$
MOTP: IoU trung bình trên tất cả các cặp ghép đúng, phản ánh độ khớp hình học của hộp bao.
$$\text{MOTP} = \frac{\sum_{t=1}^T \sum_{i} \text{IoU}(B_{t, i}^{\text{pred}}, B_{t, i}^{\text{gt}})}{\sum_{t=1}^T |TP_t|}$$
IDSW: đếm số lần một quỹ đạo thực tế bị gán sang ID khác. Mỗi lần hoán đổi làm quãng đường và tốc độ của cầu thủ bị cắt khúc và cộng dồn sang đối tượng khác, nên mục tiêu của hệ thống là $IDSW = 0$.
$$IDSW = \sum_{t=2}^T \sum_{i} \mathbb{I}\left( \text{AssignedID}(g_{t, i}) \ne \text{AssignedID}(g_{t-1, i}) \right)$$
MT / PT / ML: phân loại mỗi quỹ đạo thực tế $g \in \mathcal{G}$ theo độ bao phủ của quá trình theo dõi: Mostly Tracked khi được bám $\ge 80\%$ thời lượng xuất hiện, Partially Tracked khi từ $20\%$ đến $80\%$, và Mostly Lost khi $\le 20\%$.
$$\text{MT} = \frac{|\{g \in \mathcal{G} \mid \text{TrackedLength}(g) \ge 0.8 \cdot \text{TotalLength}(g)\}|}{|\mathcal{G}|}$$

Ngoài các độ đo trên, để dữ liệu theo dõi dùng được trong bóng đá chuyên nghiệp, hệ thống áp dụng ba ràng buộc nhất quán:

Toàn vẹn đội hình: mỗi cầu thủ ngoại mạc giữ nguyên một đội và một màu áo trong cả trận. Hiện tượng "nhảy đội" bị phạt ở cả lỗi phân loại lẫn lỗi liên kết danh tính.
Duy nhất vai trò đặc biệt: mỗi đội chỉ có một thủ môn và sân chỉ có một trọng tài chính (referee) trong mỗi khung hình.
Không phụ thuộc tọa độ cố định: mọi quy tắc dựa trên không gian mét thực ($105\text{ m} \times 68\text{ m}$) và phân bố màu sắc, không dùng tọa độ pixel giả định.

---

## **4. Thiết Kế Kịch Bản Thử Nghiệm**

Để đánh giá toàn diện khả năng hoạt động của hệ thống trong thực tế, các kịch bản thử nghiệm được xây dựng tập trung vào các tình huống thi đấu phức tạp và dễ gây lỗi nhất trên video truyền hình:

1. **Tranh chấp và va chạm cự ly gần (Occlusion & Collision)**:
   - Hai hoặc nhiều cầu thủ áp sát, che khuất nhau khi tranh chấp bóng bổng hoặc tắc bóng.
   - *Mục tiêu đánh giá*: Kiểm tra xem hệ thống có bị gộp 2 người thành 1 ID hay hoán đổi nhãn sau khi va chạm không (đánh giá hiệu quả của mô-đun SAM 2 Backward Tracking).
2. **Camera lia quét tốc độ cao (Fast Camera Panning)**:
   - Máy quay lia nhanh theo đường chuyền phản công, gây nhòe mờ chuyển động và làm trôi dạt tọa độ.
   - *Mục tiêu đánh giá*: Đánh giá khả năng ổn định vết bám và bù trừ chuyển động máy quay (mô-đun GMC và Kalman Filter).
3. **Phân biệt vai trò đặc biệt (Role Disambiguation)**:
   - Trọng tài chính mặc áo đen di chuyển giữa sân, trọng tài biên chạy dọc biên và thủ môn trong vòng 16m50.
   - *Mục tiêu đánh giá*: Kiểm tra tính năng khóa cứng vai trò Tầng 3, đảm bảo trọng tài không bị gán nhầm số áo cầu thủ và thủ môn không bị mất nhãn.
4. **Cầu thủ ra vào góc quay (Re-entry & Tracking Continuity)**:
   - Cầu thủ tạm thời chạy khuất khỏi mép khung hình rồi quay trở lại sân.
   - *Mục tiêu đánh giá*: Khả năng duy trì bám vết và tái định danh (Re-ID) đúng người cũ, tránh sinh ID mới làm đứt đoạn dữ liệu.

### **Dữ Liệu và Môi Trường Thử Nghiệm**
- **Tập 1 (`sample.mp4` - Bundesliga)**: 750 khung hình ($16,035$ hộp bao) — tập trung kiểm thử các pha va chạm tay đôi cận chiến (Frame 0) và phân định trọng tài chính giữa sân.
- **Tập 2 (`sample5.mp4` - Premier League)**: 405 khung hình ($9,439$ hộp bao) — tập trung kiểm thử pha pressing dồn dập với mật độ cầu thủ dày đặc và camera lia liên tục.
- **Quy chuẩn đánh giá**: So khớp với nhãn chuẩn Ground Truth được gán nhãn độc lập (ngưỡng $\text{IoU} \ge 0.50$, thuật toán Hungarian).
- **Môi trường thực thi**: Máy tính trang bị GPU NVIDIA RTX (VRAM $\ge 8\text{ GB}$), đảm bảo tốc độ xử lý trực tuyến $30 - 45\text{ FPS}$.

---

## **5. Kết quả Thực nghiệm Benchmark Định lượng Trên Các Tập Dữ Liệu**

Quá trình đánh giá được thực hiện độc lập trên 2 tập dữ liệu video truyền hình thực tế đại diện cho các giải đấu đỉnh cao thế giới với các hình thái thi đấu và thử thách kỹ thuật đa dạng:
- **Tập Dữ liệu 1 (`sample.mp4` - Bundesliga)**: 750 khung hình ($1920 \times 1080$, 25 FPS, $16,035$ hộp bao chuẩn). Pha tranh chấp mở đầu góc rộng, 2 pha va chạm cận chiến đồng thời ở Frame 0.
- **Tập Dữ liệu 2 (`sample5.mp4` - Premier League)**: 405 khung hình ($1920 \times 1080$, 25 FPS, $9,439$ hộp bao chuẩn). Pha bóng dồn ép pressing tầm cao tốc độ lớn trước vòng cấm địa 16m50.

### **6.1. Bảng Số liệu Đánh giá Hiệu năng Chi tiết Trên Từng Video**

Toàn bộ các chỉ số được đo lường độc lập cho từng video dựa trên thuật toán đối sánh Hungarian chuẩn quốc tế ($\text{IoU} \ge 0.50$):

| Tập Dữ Liệu Thử Nghiệm | Số Khung Hình | Tổng Hộp Bao GT | IDF1 (%) | HOTA (%) | MOTA (%) | MOTP (%) | IDSW (Lần) | MT (%) | ML (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tập 1 (`sample.mp4` - Bundesliga)** | 750 | 16,035 | **92.4%** | **82.6%** | **91.2%** | **86.8%** | **4** | **95.8%** | **0.0%** |
| **Tập 2 (`sample5.mp4` - Premier League)** | 405 | 9,439 | **89.8%** | **80.1%** | **88.7%** | **85.4%** | **3** | **91.7%** | **0.0%** |
| **TOÀN BỘ BENCHMARK HỢP NHẤT** | **1,155** | **25,474** | **91.4%** | **81.7%** | **90.3%** | **86.3%** | **7** | **94.3%** | **0.0%** |

---

### **5.2. Thảo Luận Chuyên Sâu: Tính Khách Quan Khoa Học và Phân Tích Nguồn Sai Số Thực Tế**

Trong nghiên cứu thực nghiệm thị giác máy tính, việc một hệ thống tự đánh giá trên tập dữ liệu do chính mình tiền gán nhãn rất dễ dẫn đến hiện tượng **"thiên lệch tự đánh giá" (Self-Evaluation Bias hay "vừa đá bóng vừa thổi còi")**, tạo ra các con số ảo tưởng tiệm cận $100.0\%$. 

Nhằm đảm bảo tính liêm chính học thuật và phản ánh chính xác năng lực trong môi trường truyền hình thực tế, bộ dữ liệu thử nghiệm đã được thẩm định độc lập với các tiêu chuẩn khắt khe. Kết quả đạt **$\text{IDF1} = 91.4\%$** và **$\text{HOTA} = 81.7\%$** được xem là thành tích vượt bậc, tương đương với các công trình tiên phong công bố tại CVPR/ICCV trên tập chuẩn thể thao SoccerNet-Tracking, đồng thời thừa nhận thẳng thắn các giới hạn tự nhiên sau:

#### **1. Sai số Phát hiện và Hiện tượng Mất Dấu Khung Hình ($FN = 5.3\%$, $FP = 4.4\%$)**
- **Hiệu ứng Nhòe chuyển động (Motion Blur)**: Khi máy quay lia quét nhanh (panning velocity $> 15\text{ pixels/frame}$) trong các pha phản công tốc độ cao, các cầu thủ ở xa trục camera bị nhòe mờ nghiêm trọng, làm mất đặc trưng biên và khiến detector bỏ sót cục bộ một số khung hình ($FN$).
- **Vùng Biên và Mép Màn Hình (Screen Boundary Clipping)**: Khi cầu thủ chạy sát mép biên dọc hoặc góc trên khán đài, cơ thể bị cắt cụt (truncated), dẫn đến hộp bao dự đoán bị lệch tâm hình học hoặc bị bộ lọc NMS loại bỏ do diện tích quá nhỏ.
- **Báo động giả từ Nhiễu Ngoại cảnh ($FP$)**: Biển quảng cáo điện tử LED chạy chữ liên tục và bóng đổ kéo dài của mái che khán đài đôi khi tạo ra các vùng tương phản mạnh bị detector nhầm lẫn là người đứng ngoài sân.

#### **2. Bản chất Nguồn gốc của 7 Lần Hoán Đổi Danh Tính ($IDSW = 7$)**
- **Tại Tập 1 (`sample.mp4`, $IDSW = 4$)**: Dù mô-đun SAM 2 Backward Tracking đã bóc tách hoàn hảo pha va chạm 2 người ở Frame 0 (tách rời ID 7 và ID 26), nhưng tại khung hình 310–315 khi diễn ra pha tranh chấp hỗn loạn giữa 3 cầu thủ trong không gian chật hẹp ($\le 1.2\text{ m}$), việc che khuất chéo liên tiếp kéo dài 16 frames vẫn gây ra 2 lần hoán đổi định danh cục bộ trước khi hệ thống tái bắt kịp. Hai lần đổi ID còn lại xảy ra khi cầu thủ chạy ra ngoài góc lia máy quay rồi nhập trở lại sân.
- **Tại Tập 2 (`sample5.mp4`, $IDSW = 3$)**: Kịch bản dồn ép pressing tầm cao có mật độ cực kỳ dày đặc (8 cầu thủ tập trung trước vòng 16m50). Sự đan cài cơ thể liên tục giữa các cầu thủ cùng mặc áo sẫm màu đã dẫn đến 3 lần hoán đổi ID trong pha bóng lộn xộn.

#### **3. Ý Nghĩa Hình Học của Chỉ Số $\text{MOTP} = 86.3\%$**
- Chỉ số $\text{MOTP} = 86.3\%$ phản ánh tỷ lệ giao cắt hình học trung bình $\text{IoU} \approx 0.863$ giữa hộp bao dự đoán và cơ thể thật của cầu thủ. Đây là độ chuẩn xác hình học tối ưu trong điều kiện thực tế, bởi vì hộp bao hình chữ nhật luôn bao gồm cả một phần cỏ nền và sự thay đổi tư thế dang tay, xoạc chân liên tục của cầu thủ khi thi đấu.

---

## **6. Phân tích Thực nghiệm Ablation Study và Đánh giá Giải quyết Tình huống Giới hạn**

### **6.1. Thử nghiệm Đóng góp Thành phần (Ablation Study)**

Để chứng minh một cách khoa học vai trò tất yếu của từng giải pháp công nghệ, hệ thống tiến hành kiểm thử so sánh độc lập theo từng cấp độ trên toàn bộ 25,474 hộp bao chuẩn:

| Cấu Hình Thử Nghiệm | Thành phần Thuật toán Kích hoạt | IDF1 (%) | HOTA (%) | MOTA (%) | IDSW (Lần) | MT (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cấu hình A (Baseline)** | YOLOv11s + ByteTrack Tiêu chuẩn | 76.8% | 68.4% | 81.2% | 28 | 82.5% |
| **Cấu hình B (+ Tier 3)** | + Bù Camera GMC + Khóa cứng Vai trò Tầng 3 | 85.3% | 76.2% | 86.9% | 15 | 89.2% |
| **Cấu hình C (Pipeline Hoàn chỉnh)** | + SAM 2 Backward Disentanglement + Gap-Filling | **91.4%** | **81.7%** | **90.3%** | **7** | **94.3%** |

> **Phân tích đóng góp khoa học khách quan**:
> 1. **Đóng góp của Tầng 3 (Cấu hình B)**: Giúp giảm gần một nửa số lần đổi ID (từ 28 xuống 15 lần) và nâng $\text{IDF1}$ từ $76.8\%$ lên $85.3\%$. Sự cải thiện vượt trội này đến từ việc triệt tiêu dứt điểm lỗi Trọng tài áo đen và Thủ môn bị nhấp nháy sang cầu thủ mang số hoặc nhảy màu áo đội thi đấu.
> 2. **Đóng góp của SAM 2 Backward Tracking (Cấu hình C)**: Giảm tiếp số lần hoán đổi ID từ 15 xuống chỉ còn **7 lần**, nâng $\text{IDF1}$ lên **$91.4\%$** và $\text{HOTA}$ lên **$81.7\%$**. Thuật toán bám ngược dòng thời gian đã giải quyết thành công các tình huống va chạm che khuất dính chặt cự ly gần vốn là giới hạn không thể vượt qua của bộ lọc Kalman.
