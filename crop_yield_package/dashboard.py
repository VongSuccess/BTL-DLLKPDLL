import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import plotly.express as px
from streamlit_option_menu import option_menu

# ===== CẤU HÌNH =====
st.set_page_config(
    page_title="🌾 Dashboard Năng Suất Cây Trồng",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_PATH = Path("outputs")

# ===== STYLE =====
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #eef2f3, #ffffff);
}
</style>
""", unsafe_allow_html=True)

# ===== LOAD DATA =====
def load_csv(name):
    path = DATA_PATH / name
    if path.exists():
        return pd.read_csv(path)
    return None

# ===== MENU =====
selected = option_menu(
    menu_title=None,
    options=[
        "Tổng quan",
        "Phân tích dữ liệu",
        "Hồi quy",
        "Phân loại",
        "Phân cụm",
        "Luật kết hợp"
    ],
    icons=["house", "bar-chart", "graph-up", "cpu", "diagram-3", "link"],
    orientation="horizontal",
)

# ===== HEADER =====
st.title("🌾 Hệ thống phân tích năng suất cây trồng")
st.caption("Ứng dụng Machine Learning & Data Science")

# ================= TỔNG QUAN =================
if selected == "Tổng quan":
    df = load_csv("data_overview.csv")

    if df is not None:
        col1, col2, col3 = st.columns(3)

        col1.metric("📊 Số cột", len(df))
        col2.metric("❗ Giá trị thiếu", df["missing"].sum())
        col3.metric("🔢 Tổng giá trị khác nhau", df["nunique"].sum())

        st.markdown("### 📋 Bảng dữ liệu")
        st.dataframe(df, use_container_width=True)

    summary = DATA_PATH / "summary.txt"
    if summary.exists():
        st.markdown("### 🧠 Tóm tắt kết quả")
        st.code(summary.read_text())

# ================= EDA =================
elif selected == "Phân tích dữ liệu":
    st.subheader("📊 Trực quan dữ liệu")

    col1, col2 = st.columns(2)

    if (DATA_PATH / "yield_distribution.png").exists():
        col1.image(DATA_PATH / "yield_distribution.png", caption="Phân phối năng suất")

    if (DATA_PATH / "yield_trend_by_year.png").exists():
        col2.image(DATA_PATH / "yield_trend_by_year.png", caption="Xu hướng theo năm")

    st.markdown("### 🌱 Top cây trồng / khu vực")
    for file in ["top_item_mean_yield.png", "top_area_mean_yield.png"]:
        path = DATA_PATH / file
        if path.exists():
            st.image(path)

# ================= HỒI QUY =================
elif selected == "Hồi quy":
    st.subheader("📈 So sánh mô hình hồi quy")

    df = load_csv("regression_results.csv")
    if df is not None:
        fig = px.bar(
            df,
            x="model",
            y="RMSE",
            color="model",
            title="So sánh mô hình (RMSE)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df)

    col1, col2 = st.columns(2)

    if (DATA_PATH / "actual_vs_predicted.png").exists():
        col1.image(DATA_PATH / "actual_vs_predicted.png", caption="Thực tế vs Dự đoán")

    if (DATA_PATH / "residual_plot.png").exists():
        col2.image(DATA_PATH / "residual_plot.png", caption="Sai số (Residual)")

# ================= PHÂN LOẠI =================
elif selected == "Phân loại":
    st.subheader("🧠 So sánh mô hình phân loại")

    df = load_csv("classification_results.csv")
    if df is not None:
        fig = px.bar(
            df,
            x="model",
            y="F1_macro",
            color="model",
            title="F1-score các mô hình"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df)

    col1, col2 = st.columns(2)

    if (DATA_PATH / "classification_model_comparison.png").exists():
        col1.image(DATA_PATH / "classification_model_comparison.png")

    if (DATA_PATH / "confusion_matrix.png").exists():
        col2.image(DATA_PATH / "confusion_matrix.png")

# ================= PHÂN CỤM =================
elif selected == "Phân cụm":
    st.subheader("🔍 Phân cụm dữ liệu")

    df = load_csv("cluster_profile_yield.csv")
    if df is not None:
        fig = px.bar(
            df,
            x="cluster",
            y="mean",
            title="Năng suất trung bình theo cụm"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df)

    col1, col2 = st.columns(2)

    if (DATA_PATH / "cluster_mean_yield.png").exists():
        col1.image(DATA_PATH / "cluster_mean_yield.png")

    if (DATA_PATH / "clusters_pca.png").exists():
        col2.image(DATA_PATH / "clusters_pca.png")

# ================= LUẬT KẾT HỢP =================
elif selected == "Luật kết hợp":
    st.subheader("🔗 Phân tích luật kết hợp")

    df = load_csv("top_rules_fpgrowth.csv")
    if df is not None:
        st.dataframe(df)

    if (DATA_PATH / "top_rules_lift.png").exists():
        st.image(DATA_PATH / "top_rules_lift.png")

    rec = DATA_PATH / "recommendations.txt"
    if rec.exists():
        st.markdown("### 📌 Khuyến nghị")
        st.code(rec.read_text())