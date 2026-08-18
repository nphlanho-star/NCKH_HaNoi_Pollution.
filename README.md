# 🌫️ Hanoi Air Pollution Prediction & Analysis — Nghiên Cứu & Dự Báo Ô Nhiễm Không Khí Hà Nội

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-111111?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

> **Đề Tài Nghiên Cứu Khoa Học (NCKH):** Nghiên cứu phân tích chuỗi thời gian (Time-Series Analysis), biến động không gian - thời gian (Spatial-Temporal) và áp dụng các thuật toán Học máy / Học sâu (Machine Learning / Deep Learning) để dự báo chỉ số chất lượng không khí (AQI) và nồng độ bụi mịn ($PM_{2.5}$) tại khu vực Hà Nội.

---

## 📋 Mục Lục

1. [📌 Giới Thiệu & Tính Năng Nổi Bật](#gioi-thieu)
2. [🏗️ Kiến Trúc Hệ Thống & Luồng Xử Lý](#kien-truc)
3. [📊 Bộ Dữ Liệu Sử Dụng](#du-lieu)
4. [🛠️ Công Nghệ & Thư Viện Sử Dụng](#cong-nghe)
5. [📁 Cấu Trúc Thư Mục Dự Án](#cau-truc)
6. [⚙️ Thuật Toán & Phương Pháp Chi Tiết](#thuat-toan)
7. [🚀 Hướng Dẫn Cài Đặt & Vận Hành](#cai-dat)
8. [📈 Đánh Giá Hiệu Năng Mô Hình](#danh-gia)
9. [🗺️ Định Hướng Phát Triển](#dinh-huong)

---

<a id="gioi-thieu"></a>
## 1. 📌 Giới Thiệu & Tính Năng Nổi Bật

Tình trạng ô nhiễm không khí tại Thủ đô Hà Nội ngày càng diễn biến phức tạp, đặc biệt là nồng độ bụi mịn $PM_{2.5}$ vào mùa đông. Dự án NCKH này tập trung khai thác dữ liệu từ các trạm quan trắc môi trường và yếu tố khí tượng nhằm cung cấp giải pháp dự báo sớm:

- 📊 **Phân tích khám phá dữ liệu (EDA):** Trực quan hóa xu hướng ô nhiễm theo giờ, ngày, tháng và mùa trong năm.
- 🌡️ **Kết hợp yếu tố khí tượng:** Đánh giá mối tương quan giữa nồng độ chất ô nhiễm với nhiệt độ, độ ẩm, hướng gió và tốc độ gió.
- 🔮 **Dự báo chuỗi thời gian (Time-Series Forecasting):** Dự đoán chỉ số AQI và nồng độ $PM_{2.5}$ trong 24h - 72h tới.
- 🗺️ **Bản đồ nhiệt ô nhiễm (Spatial Map):** Trực quan hóa mức độ phân bố ô nhiễm giữa các quận/huyện tại Hà Nội bằng `Folium`.
- 🖥️ **Web Dashboard tương tác:** Giao diện **Streamlit** cho phép theo dõi, tra cứu và nhận cảnh báo mức độ độc hại không khí thời gian thực.

---

<a id="kien-truc"></a>
## 2. 🏗️ Kiến Trúc Hệ Thống & Luồng Xử Lý

```text
┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│  Trạm Quan Trắc & Weather│ ──> │ Tiền xử lý & Làm sạch  │ ──> │ Trích xuất Đặc trưng   │
│  (PM2.5, AQI, Temp,...) │     │ (Missing, Outliers)    │     │ (Lag, Rolling, Weather)│
└────────────────────────┘     └────────────────────────┘     └────────────────────────┘
                                                                           │
                                                                           ▼
┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
│  Streamlit Dashboard   │ <── │ Cảnh Báo Mức Độ AQI    │ <── │ Huấn Luyện Mô Hình     │
│  (Dự báo & Trực quan)  │     │ (Phân cấp theo WHO)    │     │ (XGBoost / LSTM/...)   │
└────────────────────────┘     └────────────────────────┘     └────────────────────────┘
```

---

<a id="du-lieu"></a>
## 3. 📊 Bộ Dữ Liệu Sử Dụng

Dữ liệu được thu thập từ các trạm quan trắc không khí (PAM Air, Cổng thông tin quan trắc Hà Nội, U.S. Embassy Air Quality Monitor) kết hợp dữ liệu khí tượng OpenWeatherMap:

| Nhóm Dữ Liệu | Chỉ Số / Trường Thông Tin | Đơn Vị Đo | Ý Nghĩa Môi Trường |
| :--- | :--- | :---: | :--- |
| **Chất Ô Nhiễm** | `PM2.5`, `PM10` | $\mu g/m^3$ | Bụi mịn và bụi siêu mịn nguy hại phổi |
| **Ký Hiệu Hóa Học** | `CO`, `NO2`, `SO2`, `O3` | $ppb$ / $ppm$ | Khí thải từ phương tiện giao thông & công nghiệp |
| **Khí Tượng** | `Temperature`, `Humidity` | $^\circ C$, $\%$ | Nhiệt độ và độ ẩm ảnh hưởng tới sự khuếch tán bụi |
| **Gió & Áp Suất** | `Wind Speed`, `Pressure` | $m/s$, $hPa$ | Tốc độ gió và áp suất khí quyển |
| **Đầu Ra Target** | `AQI_US`, `AQI_VN` | Chỉ số (0 - 500) | Chỉ số chất lượng không khí tổng hợp |

---

<a id="cong-nghe"></a>
## 4. 🛠️ Công Nghệ & Thư Viện Sử Dụng

- **Ngôn ngữ lập trình:** `Python 3.8+`
- **Thao tác & Xử lý Dữ liệu:** `Pandas`, `NumPy`, `SciPy`
- **Trực quan hóa & Bản đồ:** `Matplotlib`, `Seaborn`, `Plotly`, `Folium`
- **Mô Hình Học Máy & Chuỗi Thời Gian:**
  - `Scikit-Learn` (Random Forest, Ridge, StandardScaler)
  - `XGBoost`, `LightGBM` (Gradient Boosting)
  - `Statsmodels` (ARIMA, SARIMAX)
  - `PyTorch` / `TensorFlow` (LSTM, GRU cho Deep Learning)
- **Giao diện Web App:** `Streamlit`
- **Môi trường Phát triển:** VS Code / Jupyter Notebook / Google Colab

---

<a id="cau-truc"></a>
## 5. 📁 Cấu Trúc Thư Mục Dự Án

DeCuongMonHocMovieLens/
│
├── data/                       # Dữ liệu dự án
│   ├── raw/                    # Dữ liệu chất lượng không khí gốc chưa xử lý
│   └── processed/              # Dữ liệu đã làm sạch, nội suy và nội suy chuỗi thời gian
│
├── notebooks/                  # Jupyter Notebooks phân tích & thử nghiệm
│   ├── 01_Data_Cleaning_Imputation.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_TimeSeries_Feature_Engineering.ipynb
│   ├── 04_ML_Models_XGBoost_RF.ipynb
│   └── 05_LSTM_DeepLearning_Forecasting.ipynb
│
├── src/                        # Mã nguồn chính dự án
│   ├── __init__.py
│   ├── data_pipeline.py        # Pipeline nạp, làm sạch và trích xuất Lag Features
│   ├── models.py               # Lớp huấn luyện và dự báo AQI / PM2.5
│   └── utils.py                # Hàm tính toán chỉ số AQI chuẩn WHO/VN và độ đo
│
├── app.py                      # Dashboard dự báo & theo dõi ô nhiễm bằng Streamlit
├── requirements.txt            # Danh sách thư viện cần cài đặt
├── .gitignore                  # Bỏ qua các file bytecode, data nặng
└── README.md                   # Tài liệu hướng dẫn chi tiết dự án

---

<a id="thuat-toan"></a>
## 6. ⚙️ Thuật Toán & Phương Pháp Chi Tiết

### 6.1. Tiền Xử Lý Dữ Liệu Chuỗi Thời Gian
- **Xử lý dữ liệu khuyết:** Áp dụng kỹ thuật nội suy chuỗi thời gian (**Time-series Interpolation**) và **Forward Fill** cho các khoảng mất tín hiệu trạm đo.
- **Tạo đặc trưng thời gian (Lag & Rolling Features):** Tạo các biến độ trễ $t-1, t-2, t-24$ (giờ) và trung bình động (**Rolling Mean 24h**) để mô hình học tính chu kỳ.

### 6.2. Thuật Toán Dự Báo Gradient Boosting (XGBoost / LightGBM)
- Coi bài toán dự báo chuỗi thời gian là bài toán Học có giám sát (Supervised Regression).
- Sử dụng hàm tối ưu hóa Loss Function bằng Gradient Descent trên cây quyết định:
  $$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

### 6.3. Học Sâu Chuỗi Thời Gian (LSTM - Long Short-Term Memory)
- Áp dụng mạng Nơ-ron hồi quy **LSTM** để ghi nhớ các phụ thuộc chuỗi thời gian dài hạn (Long-term dependencies) trong nồng độ bụi $PM_{2.5}$, khắc phục hiện tượng biến mất đạo hàm (Vanishing Gradient).

---

<a id="cai-dat"></a>
## 7. 🚀 Hướng Dẫn Cài Đặt & Vận Hành

### Bước 1: Clone Repository
```bash
git clone [https://github.com/nphlanho-star/NCKH_HaNoi_Pollution.git](https://github.com/nphlanho-star/NCKH_HaNoi_Pollution.git)
cd NCKH_HaNoi_Pollution
```

### Bước 2: Khởi Tạo & Kích Hoạt Môi Trường Ảo
```bash
# Đối với Windows
python -m venv venv
venv\Scripts\activate

# Đối với macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài Đặt Thư Viện Phụ Thuộc
```bash
pip install -r requirements.txt
```

### Bước 4: Khởi Chạy Dashboard Dự Báo (Streamlit)
```bash
streamlit run app.py
```
> Trình duyệt sẽ tự động mở trang web dashboard tại địa chỉ: `http://localhost:8501`

---

<a id="danh-gia"></a>
## 8. 📈 Đánh Giá Hiệu Năng Mô Hình

Các mô hình được kiểm thử dự báo nồng độ $PM_{2.5}$ trước 24 giờ trên tập dữ liệu Test với các chỉ số đo lường chính:

- **RMSE (Root Mean Squared Error):** Độ lệch bình phương trung bình.
- **MAE (Mean Absolute Error):** Độ lệch tuyệt đối trung bình.
- **$R^2$ Score:** Hệ số xác định thể hiện mức độ giải thích biến thiên của mô hình.

| Mô Hình (Algorithm) | RMSE ($\mu g/m^3$) | MAE ($\mu g/m^3$) | $R^2$ Score | Đánh Giá Thực Tế |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 18.42 | 13.15 | 0.62 | Mô hình Baseline, chưa bắt được phi tuyến |
| **ARIMA / SARIMAX** | 15.80 | 11.20 | 0.71 | Dự báo ngắn hạn tốt, yếu khi có biến động khí tượng đột ngột |
| **Random Forest Regressor**| 12.35 | 8.65 | 0.82 | Bắt tốt các mối quan hệ phi tuyến |
| **XGBoost Regressor** | **9.85** | **6.40** | **0.89** | **Tối ưu nhất, tốc độ xử lý nhanh và độ chính xác cao** |
| **LSTM (Deep Learning)** | **10.12** | **6.85** | **0.88** | Khả năng học chuỗi thời gian dài rất tốt |

---

<a id="dinh-huong"></a>
## 9. 🗺️ Định Hướng Phát Triển

- [ ] **Tích hợp cảm biến IoT thời gian thực:** Kết nối API thu thập dữ liệu trực tiếp từ các trạm quan trắc giá rẻ cá nhân.
- [ ] **Mô hình Spatial-Temporal (ST-GCN):** Áp dụng Graph Convolutional Network để dự báo đồng thời không gian lẫn thời gian trên toàn địa bàn Hà Nội.
- [ ] **Hệ thống cảnh báo tự động:** Tích hợp Bot Telegram / Zalo gửi tin nhắn cảnh báo khi AQI vượt ngưỡng nguy hại (AQI > 200).
- [ ] **Đóng gói Docker & Cloud Deploy:** Đóng gói ứng dụng thành Docker Container và triển khai lên AWS/GCP.

---
<p align="center">
  <i>⭐ Đừng quên tặng 1 Star trên GitHub nếu bạn thấy đề tài nghiên cứu này hữu ích! ⭐</i>
</p>
