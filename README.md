# He-thong-DSS-du-bao-san-xuat-trong-nong-nghiep
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io

# Đọc dữ liệu
file_path = "data_mau.csv"
df = pd.read_csv(file_path)

# Tách dữ liệu thành biến đầu vào (X) và biến mục tiêu (y)
X = df.drop(columns=["Năng suất dự báo (tấn/ha)"])
y = df["Năng suất dự báo (tấn/ha)"]

# Xử lý biến danh mục
categorical_cols = ["Loại cây trồng", "Loại đất", "Kỹ thuật canh tác"]
encoder = OneHotEncoder(drop="first", sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
X_encoded.columns = encoder.get_feature_names_out(categorical_cols)

# Kết hợp với các biến số
X_numeric = X.drop(columns=categorical_cols).reset_index(drop=True)
X_processed = pd.concat([X_numeric, X_encoded], axis=1)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Giao diện Streamlit
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stApp {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 20px;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .stDownloadButton {
            text-align: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Dự báo năng suất cây trồng</h1>", unsafe_allow_html=True)
st.markdown("---")

# Chọn loại cây trồng, loại đất, và kỹ thuật canh tác
st.markdown("### 🌱 Nhập thông tin cây trồng:")
col1, col2, col3 = st.columns(3)
with col1:
    crop_options = df["Loại cây trồng"].unique()
    selected_crop = st.selectbox("Loại cây trồng", crop_options)
with col2:
    soil_options = df["Loại đất"].unique()
    selected_soil = st.selectbox("Loại đất", soil_options)
with col3:
    technology_options = df["Kỹ thuật canh tác"].unique()
    selected_technology = st.selectbox("Kỹ thuật canh tác", technology_options)

# Nhập dữ liệu thực tế
st.markdown("### ☀️ Điều kiện môi trường:")
col1, col2 = st.columns(2)
with col1:
    temp = st.number_input("🌡️ Nhiệt độ (°C)", value=25.0)
    rain = st.number_input("🌧️ Lượng mưa (mm)", value=100.0)
with col2:
    humidity = st.number_input("💧 Độ ẩm (%)", value=60.0)
    fertilizer = st.number_input("🌿 Lượng phân bón (kg/ha)", value=50.0)

# Nút dự báo
if st.button("📊 Dự báo"):
    input_data = pd.DataFrame([[selected_crop, selected_soil, selected_technology, temp, rain, humidity, fertilizer]],
                              columns=["Loại cây trồng", "Loại đất", "Kỹ thuật canh tác", "Nhiệt độ (°C)",
                                       "Lượng mưa (mm)", "Độ ẩm (%)", "Lượng phân bón (kg/ha)"])

    # Xử lý biến danh mục
    input_encoded = pd.DataFrame(0, index=[0], columns=encoder.get_feature_names_out(categorical_cols))
    for col, value in zip(["Loại cây trồng", "Loại đất", "Kỹ thuật canh tác"],
                          [selected_crop, selected_soil, selected_technology]):
        col_name = f"{col}_{value}"
        if col_name in input_encoded.columns:
            input_encoded[col_name] = 1

    input_processed = pd.concat([input_data.drop(columns=categorical_cols), input_encoded], axis=1)
    input_processed = input_processed[X_processed.columns]
    input_scaled = scaler.transform(input_processed)

    # Dự đoán năng suất
    predicted_yield = model.predict(input_scaled)[0]
    st.success(f"🌾 **Năng suất dự báo:** {predicted_yield:.2f} tấn/ha")

    # Khuyến nghị
    st.markdown("### 📢 Khuyến nghị:")
    if predicted_yield < 12:
        st.error("⚠️ **Năng suất thấp**. Cần tăng cường lượng phân bón và áp dụng kỹ thuật canh tác tiên tiến.")
    elif 12 <= predicted_yield < 22:
        st.warning("🔎 **Năng suất trung bình**. Nên tối ưu hóa lượng nước và dinh dưỡng trong đất.")
    else:
        st.success("✅ **Năng suất cao**! Hãy tiếp tục duy trì kỹ thuật canh tác hiện tại.")

    # Tạo báo cáo dữ liệu
    data_report = pd.DataFrame({
        "Loại cây trồng": [selected_crop],
        "Loại đất": [selected_soil],
        "Kỹ thuật canh tác": [selected_technology],
        "Nhiệt độ (°C)": [temp],
        "Lượng mưa (mm)": [rain],
        "Độ ẩm (%)": [humidity],
        "Lượng phân bón (kg/ha)": [fertilizer],
        "Năng suất dự báo (tấn/ha)": [predicted_yield]
    })

    st.markdown("### 📊 Báo cáo năng suất:")
    st.dataframe(data_report)

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=["Dự báo"], y=[predicted_yield], palette="viridis", ax=ax)
    ax.set_ylabel("Năng suất (tấn/ha)")
    ax.set_title("📊 Biểu đồ năng suất dự báo")
    st.pyplot(fig)

    # Xuất file Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        data_report.to_excel(writer, index=False)
    st.download_button(
        label="📥 Tải xuống báo cáo",
        data=output.getvalue(),
        file_name="data_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


