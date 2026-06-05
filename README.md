# 📊 Dự Án Xử Lý Dữ Liệu Bị Thiếu (Missing Values Handling)

## 📋 Mục Đích Dự Án

Dự án này tập trung vào **nghiên cứu và thực hành các phương pháp xử lý dữ liệu bị thiếu (Missing Values)** trong Machine Learning.
Bao gồm:

- ✅ Thử nghiệm nhiều kỹ thuật imputation (KNN, Iterative, MICE)
- ✅ Tạo dữ liệu tổng hợp với các cơ chế thiếu khác nhau (MAR, MCAR, MNAR)
- ✅ Ứng dụng trên dữ liệu thực tế (Horse Colic, Titanic, World Bank)
- ✅ Tham gia các cuộc thi Kaggle (Tabular Playground, WiDS Datathon)

---

## 📁 Cấu Trúc Thư Mục

### 1. 📦 `data/` - Kho Dữ Liệu

Chứa tất cả bộ dữ liệu sử dụng trong dự án, bao gồm dữ liệu thực tế và tổng hợp.

#### **A. `data/real data/` - Dữ Liệu Thực Tế**

| File                    | Mô Tả                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| `horse_colic_clean.csv` | Dataset về bệnh xoắn ruột ngựa - **dữ liệu đã làm sạch**                                    |
| `horse_colic_train.csv` | Dữ liệu huấn luyện cho bài toán phân loại bệnh ngựa                                         |
| `horse_colic_test.csv`  | Dữ liệu kiểm tra cho bài toán phân loại bệnh ngựa                                           |
| `Titanic-Dataset.csv`   | Dataset nổi tiếng về hành khách tàu Titanic - **có nhiều dữ liệu thiếu**                    |
| `world_bank.csv`        | Dữ liệu kinh tế và xã hội từ Ngân hàng Thế giới - **phục vụ phân tích xu hướng phát triển** |

**Số file**: 5 CSV  
**Mục đích**: Cơ sở để thử nghiệm các phương pháp xử lý dữ liệu thiếu trên dữ liệu thực

---

#### **B. `data/synthetic data/` - Dữ Liệu Tổng Hợp**

Chứa dữ liệu tổng hợp được tạo với các cơ chế thiếu **có chủ ý và có kiểm soát** để thử nghiệm.

##### **📌 `data/synthetic data/MCAR/` - Missing Completely At Random**

| File                  | Mô Tả                                                           |
| --------------------- | --------------------------------------------------------------- |
| `dataset_goc.csv`     | Dữ liệu gốc hoàn chỉnh (không có dữ liệu thiếu)                 |
| `dataset_missing.csv` | Dữ liệu sau khi thêm các giá trị thiếu **hoàn toàn ngẫu nhiên** |
| `MCAR.ipynb`          | Notebook giải thích cơ chế MCAR và quá trình tạo dữ liệu        |

**Cơ chế**: Dữ liệu bị thiếu một cách hoàn toàn ngẫu nhiên, không phụ thuộc vào bất kỳ biến nào.

---

##### **📌 `data/synthetic data/MAR/` - Missing At Random**

| File                  | Mô Tả                                                                               |
| --------------------- | ----------------------------------------------------------------------------------- |
| `dataset_goc.csv`     | Dữ liệu gốc hoàn chỉnh                                                              |
| `dataset_missing.csv` | Dữ liệu sau khi thêm các giá trị thiếu **có ngẫu nhiên có điều kiện**               |
| `MAR.ipynb`           | Notebook giải thích cơ chế MAR - giá trị thiếu phụ thuộc vào các biến quan sát được |

**Cơ chế**: Dữ liệu bị thiếu phụ thuộc vào các biến khác được quan sát, nhưng không phụ thuộc vào chính giá trị bị thiếu đó.

---

##### **📌 `data/synthetic data/MNAR/` - Missing Not At Random**

| File                  | Mô Tả                                                                  |
| --------------------- | ---------------------------------------------------------------------- |
| `dataset_goc.csv`     | Dữ liệu gốc hoàn chỉnh                                                 |
| `dataset_missing.csv` | Dữ liệu sau khi thêm các giá trị thiếu **có cơ chế**                   |
| `MNAR.ipynb`          | Notebook giải thích cơ chế MNAR - giá trị thiếu phụ thuộc vào chính nó |

**Cơ chế**: Dữ liệu bị thiếu phụ thuộc vào chính giá trị bị thiếu đó (nguy hiểm nhất - có bias).

---

##### **📌 `data/synthetic data/ket_hop/` - Missing Kết Hợp**

| File                  | Mô Tả                                                  |
| --------------------- | ------------------------------------------------------ |
| `dataset_goc.csv`     | Dữ liệu gốc hoàn chỉnh                                 |
| `dataset_missing.csv` | Dữ liệu tổng hợp kết hợp nhiều cơ chế thiếu            |
| `ket_hop.ipynb`       | Notebook mô tả cách kết hợp các cơ chế MAR, MCAR, MNAR |

**Cơ chế**: Sự kết hợp của nhiều cơ chế thiếu khác nhau trong cùng một dataset.

**Số file**: 12 CSV + Notebook  
**Mục đích**: Giúp hiểu rõ các cơ chế thiếu dữ liệu khác nhau và ảnh hưởng của chúng

---

### 2. 📋 `Handle missing values/` - Phương Pháp Xử Lý

Chứa các notebook thử nghiệm **các phương pháp khác nhau** để xử lý dữ liệu bị thiếu.

| File                      | Mô Tả                                                                     | Phương Pháp                        |
| ------------------------- | ------------------------------------------------------------------------- | ---------------------------------- |
| `horse_colic.ipynb`       | Thử nghiệm xử lý dữ liệu thiếu trên dataset ngựa                          | Đa phương pháp                     |
| `iterative_imputer.ipynb` | Phương pháp **Iterative Imputation** - dự đoán lặp lại cho mỗi biến       | Scikit-learn IterativeImputer      |
| `knn_imputer.ipynb`       | Phương pháp **KNN Imputation** - dùng k-nearest neighbors để điền giá trị | KNN                                |
| `mice.ipynb`              | Phương pháp **MICE** (Multivariate Imputation by Chained Equations)       | Imputation theo phương trình chuỗi |
| `world_bank.ipynb`        | Thử nghiệm xử lý dữ liệu thiếu trên dataset Ngân hàng Thế giới            | Đa phương pháp                     |

**Số file**: 5 Jupyter Notebooks  
**Mục đích**: So sánh hiệu quả các phương pháp imputation khác nhau trên các dataset thực tế

---

### 3. 🔧 `Sklearn/` - Công Cụ & Công Cụ Cải Tiến

Chứa các công cụ tái sử dụng được phát triển để cải tiến các phương pháp imputation.

| File                      | Loại          | Mô Tả                                                                                                       |
| ------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------- |
| `imputer_tools_mi.py`     | Python Module | **Công cụ chính** - Chứa các hàm: `calculate_nrmse()` tính sai số, `missing_imputer()` thực hiện imputation |
| `iterative_imputer.ipynb` | Notebook      | Thử nghiệm và so sánh Iterative Imputer                                                                     |
| `missing_indicator.ipynb` | Notebook      | Phân tích **Missing Indicator** - tạo biến chỉ báo cho giá trị thiếu                                        |

**Số file**: 3 file  
**Mục đích**: Tổ chức lại code và công cụ để tái sử dụng, hỗ trợ tính toán metric hiệu quả

---

### 4. 🎮 `Tabular Playground Series - Jun 2022/` - Cuộc Thi Kaggle

Dự án tham gia cuộc thi **Kaggle Tabular Playground Series** tháng 6/2022.

| File        | Mô Tả                                                                     |
| ----------- | ------------------------------------------------------------------------- |
| `data.csv`  | Dữ liệu cuộc thi (bộ dữ liệu tabular)                                     |
| `eda.ipynb` | **EDA (Exploratory Data Analysis)** - Phân tích khám phá dữ liệu chi tiết |

**Số file**: 2 file  
**Mục đích**: Áp dụng các phương pháp xử lý dữ liệu thiếu vào cuộc thi thực tế

---

### 5. 🏆 `WiDS Datathon/` - Cuộc Thi Datathon

Dự án tham gia **Women in Data Science (WiDS) Datathon 2021**.

| File                         | Mô Tả                                                                 |
| ---------------------------- | --------------------------------------------------------------------- |
| `TrainingWiDS2021.csv`       | Dữ liệu huấn luyện cho cuộc thi                                       |
| `UnlabeledWiDS2021.csv`      | Dữ liệu không có nhãn (dữ liệu cần dự đoán)                           |
| `adversial_validation.ipynb` | **Adversarial Validation** - Kiểm tra độ tương tự giữa train/test set |
| `eda_top1.ipynb`             | **Phân tích EDA #1** - Tổng quát về dữ liệu                           |
| `eda_top2.ipynb`             | **Phân tích EDA #2**                                                  |
| `eda_top3.ipynb`             | **Phân tích EDA #3**                                                  |
| `eda_top4.ipynb`             | **Phân tích EDA #4**                                                  |
| `eda_top8.ipynb`             | **Phân tích EDA #8**                                                  |

**Số file**: 8 file (2 CSV + 6 Notebook)  
**Mục đích**: Phân tích dữ liệu từ cuộc thi WiDS Datathon 2021, ứng dụng các kỹ thuật học được

---

## 📊 Thống Kê Tổng Quát

| Loại                  | Số Lượng  |
| --------------------- | --------- |
| **CSV Files**         | 9 file    |
| **Jupyter Notebooks** | 15 file   |
| **Python Scripts**    | 1 file    |
| **Tổng File**         | 25+ file  |
| **Thư Mục Chính**     | 6 thư mục |

---

## 🎯 Quy Trình Làm Việc Của Dự Án

```
1. Dữ Liệu (data/)
   ↓
   ├─ Dữ liệu thực tế (Titanic, Horse Colic, World Bank)
   └─ Dữ liệu tổng hợp (MCAR, MAR, MNAR, Kết hợp)

2. Hiểu Rõ Cơ Chế Thiếu
   ↓
   └─ Phân tích từng cơ chế (MCAR, MAR, MNAR)

3. Thử Nghiệm Phương Pháp (Handle missing values/)
   ↓
   ├─ KNN Imputation
   ├─ Iterative Imputation
   └─ MICE

4. Công Cụ & Tối Ưu (Minh/)
   ↓
   └─ Tạo hàm tái sử dụng, tính NRMSE

5. Ứng Dụng Thực Tế
   ↓
   ├─ Kaggle Tabular Playground
   └─ WiDS Datathon
```

---

## 💡 Điểm Nổi Bật Của Dự Án

✅ **Cấu trúc tổ chức rõ ràng**: Từ dữ liệu → phương pháp → ứng dụng  
✅ **Có công cụ tái sử dụng**: File `imputer_tools_mi.py` với metric NRMSE  
✅ **Bao gồm cơ sở lý thuyết**: Các notebook giải thích MAR, MCAR, MNAR  
✅ **Dữ liệu đa dạng**: Kết hợp thực tế và tổng hợp  
✅ **Ứng dụng thực tế**: Tham gia các cuộc thi Kaggle  
✅ **Hỗ trợ nhiều phương pháp**: KNN, Iterative, MICE

---

## 🔗 Các Phương Pháp Imputation Chính

| Phương Pháp        | File                      | Ưu Điểm                                   | Nhược Điểm           |
| ------------------ | ------------------------- | ----------------------------------------- | -------------------- |
| **KNN Imputation** | `knn_imputer.ipynb`       | Đơn giản, không cần huấn luyện            | Chậm với dữ liệu lớn |
| **Iterative**      | `iterative_imputer.ipynb` | Linh hoạt, hỗ trợ nhiều estimator         | Phức tạp hơn         |
| **MICE**           | `mice.ipynb`              | Tối ưu thống kê, hỗ trợ nhiều phương pháp | Tính toán phức tạp   |

---

## 📌 Hướng Dẫn Sử Dụng

### Để bắt đầu:

1. Xem dữ liệu tổng hợp tại `data/synthetic data/` để hiểu các cơ chế thiếu
2. Chạy các notebook trong `Handle missing values/` để xem các phương pháp
3. Sử dụng công cụ từ `Minh/imputer_tools_mi.py` cho các dự án riêng
4. Tham khảo phân tích EDA từ cuộc thi tại `WiDS Datathon/`

### Để thêm dữ liệu mới:

- Đặt tại `data/real data/` nếu là dữ liệu thực tế
- Hoặc tạo thư mục con tại `data/synthetic data/` nếu là dữ liệu tổng hợp

---

## 👨‍💻 Công Nghệ Sử Dụng

- **Python**: Ngôn ngữ lập trình chính
- **Jupyter Notebook**: Thử nghiệm và phân tích
- **Scikit-learn**: Machine Learning library chính
- **Pandas**: Xử lý dữ liệu
- **NumPy**: Tính toán khoa học
- **Git**: Quản lý phiên bản

---

## 📝 Ghi Chú

- Tất cả dữ liệu CSV nằm trong `data/` folder
- Các notebook `.ipynb` có thể chạy độc lập
- Công cụ chung nằm tại `Sklearn/imputer_tools_mi.py`
- Mỗi cơ chế thiếu (MCAR, MAR, MNAR) có một thư mục riêng với notebook giải thích

---

**Cập nhật lần cuối**: 2026-06-05
