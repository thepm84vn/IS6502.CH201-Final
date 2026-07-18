# Ứng dụng học máy trong phân tích hành vi tương tác học tập trực tuyến và dự đoán kết quả học viên

**Nghiên cứu trên OULAD (đầy đủ) + Kiểm chứng chéo trên 3 bộ dữ liệu độc lập ở 4 khu vực địa lý**

> Chuyên đề: Phân tích dữ liệu · Trạng thái bài báo: ☐ Đã xuất bản ☐ Đã gửi ☒ Chuẩn bị gửi

---

## 1. Mô tả dự án

Dự án ứng dụng học máy để dự đoán kết quả học tập của sinh viên trong môi trường học trực tuyến. Nghiên cứu là một thiết kế **tam giác hóa (triangulation) qua 4 khu vực địa lý**, với 1 dataset chính và 3 dataset đối chứng độc lập:

| # | Dataset | Năm | Khu vực | Quy mô | Đặc điểm |
|---|---|---|---|---|---|
| 1 | **OULAD** (Kuzilek et al.) | 2013–2014 | Anh | 32.593 SV | Chính — clickstream VLE đầy đủ |
| 2 | **Dropout Academic Success** (Realinho et al.) | 2021 | Bồ Đào Nha | 4.424 SV | Đối chứng — khác thời gian/quốc gia |
| 3 | **North America Course** (Injadat et al.) | 2020 | Bắc Mỹ | 486 SV | Đối chứng — mẫu nhỏ, mất cân bằng nặng |
| 4 | **xAPI-Edu** (Amrieh et al.) | 2016 | Trung Đông | 480 SV | Đối chứng — hành vi tương tác thuần túy (gần OULAD nhất) |

### Tính mới
1. Phân tích đa mốc thời gian có hệ thống (early prediction).
2. Kết hợp Explainable AI (SHAP) với bài toán dự đoán sớm.
3. Tách bạch đóng góp đặc trưng hành vi vs. nhân khẩu học tĩnh.
4. Hàm ý triển khai cụ thể: cảnh báo sơ bộ khả thi từ giai đoạn rất sớm.
5. **Kiểm chứng chéo trên 4 khu vực địa lý** — không chỉ "xác nhận" mà phát hiện các sắc thái khoa học:
   - **xAPI-Edu**: chỉ 4 đặc trưng hành vi thuần túy (không điểm số) → AUC-ROC = 0,806 — bằng chứng trực tiếp cho luận điểm cốt lõi.
   - **Bắc Mỹ**: ưu thế ensemble bị đảo ngược trên dữ liệu mất cân bằng cực đoan → ưu thế ensemble phụ thuộc độ cân bằng dữ liệu, không phải quy luật tuyệt đối.

*(Xem chi tiết lập luận tính mới trong `cover_letter.docx`.)*

### Câu hỏi nghiên cứu
- **RQ1:** Mô hình học máy nào cho hiệu suất dự đoán tốt nhất?
- **RQ2:** Đặc trưng hành vi nào ảnh hưởng mạnh nhất đến kết quả học tập?
- **RQ3:** Có thể dự đoán sớm từ mốc thời gian nào?
- **RQ4:** Phát hiện có được tái khẳng định trên các dữ liệu độc lập hay không?

### Kết quả chính — OULAD (32.593 sinh viên)
| Mốc | Mô hình tốt nhất | Accuracy | F1-score | AUC-ROC |
|---|---|---|---|---|
| 25% | Random Forest | 0,672 | 0,666 | 0,781 |
| 50% | Random Forest | 0,726 | 0,724 | 0,841 |
| 75% | **RF ≈ XGBoost** | **≈0,770** | **≈0,77** | **≈0,88** |

### Kết quả đối chứng
| Dataset | Mốc tốt nhất | Mô hình | F1 | AUC-ROC | Ghi chú |
|---|---|---|---|---|---|
| 2021 Bồ Đào Nha | Sau HK2 | Random Forest | 0,769 | 0,903 | Tái khẳng định đầy đủ |
| 2020 Bắc Mỹ | Mốc 50% | Logistic Reg ≈ SVM | 0,884 | 0,956 | Ensemble bị đảo ngược (mất cân bằng) |
| xAPI-Edu Trung Đông | Chỉ hành vi | Random Forest | 0,652 | **0,806** | Không dùng điểm số! |
| xAPI-Edu Trung Đông | Đầy đủ | Random Forest | 0,825 | 0,924 | + đặc trưng bối cảnh |

→ Điểm chung xuyên suốt cả 4 dataset: **các chỉ số tham gia & hoàn thành nhiệm vụ học tập là tín hiệu dự đoán mạnh nhất**, bất kể bối cảnh địa lý hay nền tảng.

---

## 2. Cấu trúc thư mục

```
.
├── README.md
├── code_oulad/                     # Pipeline chính (OULAD)
│   ├── 01_preprocess.py ... 05_extra_charts.py
├── code_dropout2021/               # Đối chứng #1 (Bồ Đào Nha)
│   └── train_dropout2021.py
├── code_namerica/                  # Đối chứng #2 (Bắc Mỹ)
│   └── train_namerica.py
├── code_xapi/                      # Đối chứng #3 (Trung Đông)
│   └── train_xapi.py
└── images/                         # 10 biểu đồ kết quả
```

---

## 3. Cách chạy lại thực nghiệm

### Yêu cầu
```bash
pip install pandas numpy scikit-learn xgboost shap imbalanced-learn matplotlib seaborn joblib pyreadr --break-system-packages
```

### 3.1. OULAD (chính)
Nguồn đầy đủ (32.593 SV, 10.655.280 bản ghi VLE): repo R package `github.com/jakubkuzilek/oulad` (đọc `data/*.rda` bằng `pyreadr`), hoặc CSV chính thức tại https://analyse.kmi.open.ac.uk/open-dataset
```bash
cd code_oulad/
python3 01_preprocess.py && python3 02_eda.py && python3 03_train_models.py && python3 04_shap_analysis.py && python3 05_extra_charts.py
```

### 3.2. Đối chứng #1 — 2021 Bồ Đào Nha
Nguồn: UCI ML Repository, `doi: 10.24432/C5MC89`
```bash
cd code_dropout2021/ && python3 train_dropout2021.py
```

### 3.3. Đối chứng #2 — 2020 Bắc Mỹ
Nguồn: `github.com/Western-OC2-Lab/Student-Performance-and-Engagement-Prediction-eLearning-datasets`
```bash
cd code_namerica/ && python3 train_namerica.py
```

### 3.4. Đối chứng #3 — xAPI-Edu Trung Đông
Nguồn: Kaggle `aljarah/xAPI-Edu-Data` (Kalboard 360 LMS). File `xAPI-Edu-Data.csv` (480 records).
```bash
cd code_xapi/ && python3 train_xapi.py
```

---

## 4. Hạn chế đã biết
1. Nhóm "Withdrawn" (OULAD) chưa đưa vào bài toán phân loại chính.
2. Đặc trưng ở dạng tổng hợp tĩnh, chưa mô hình hóa chuỗi thời gian (hướng mở rộng: LSTM/GRU).
3. Bốn dataset bao phủ Anh/Bồ Đào Nha/Bắc Mỹ/Trung Đông nhưng đều là giáo dục chính quy; MOOC quy mô lớn và Đông Á/Đông Nam Á cần kiểm chứng thêm.
4. Cấu trúc mốc thời gian và quy mô mẫu khác nhau đáng kể (480 đến 32.593) — so sánh định lượng trực tiếp cần thận trọng, mang tính xu hướng.
5. Dataset Bắc Mỹ mất cân bằng cực đoan (8/486 mẫu Weak) → ước lượng AUC có độ biến thiên cao.

## 5. Việc cần hoàn thiện trước khi nộp
- [ ] Đọc & xác nhận nội dung tài liệu tham khảo [1]-[10]
- [ ] Điền thông tin học viên/GVHD vào cover letter & slide tiêu đề/kết thúc
- [ ] Kiểm tra đạo văn (Turnitin, DoIT...)
- [ ] Định dạng theo template tạp chí/hội thảo đích

## 6. Nguồn dữ liệu & Trích dẫn
1. Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171
2. Realinho, V., et al. (2021). Predict Students' Dropout and Academic Success. UCI ML Repository. https://doi.org/10.24432/C5MC89
3. Injadat, M., et al. (2020). Multi-split optimized bagging ensemble model selection for multi-class educational data mining. *Applied Intelligence*, 50, 4506–4528. https://doi.org/10.1007/s10489-020-01776-3
4. Amrieh, E. A., Hamtini, T., & Aljarah, I. (2016). Mining Educational Data to Predict Student's Academic Performance using Ensemble Methods. *Int. J. Database Theory and Application*, 9(8), 119–136. (xAPI-Edu-Data, Kalboard 360)
