import streamlit as st
import pandas as pd
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Data Mining Testbed", page_icon="🎓", layout="wide")

st.title("🎓 Higher Education Data Mining Testbed")
st.write("End-to-end pipeline on Teaching Quality dataset: Overview, Preprocessing, Modeling, Evaluation.")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Testbed Configuration")
    uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
    
    st.divider()
    st.subheader("🛠️ Data Preparation")
    use_smote = st.toggle("Aktifkan SMOTE (Balancing)", value=True)
    test_size = st.slider("Test Size (Holdout)", 0.1, 0.4, 0.2, step=0.05)
    
    st.divider()
    st.subheader("🧠 Hyperparameter Tuning")
    n_estimators = st.slider("n_estimators (Jumlah Pohon)", 50, 200, 100, step=10)
    max_depth = st.slider("max_depth (Kedalaman Maksimal)", 5, 50, 20, step=5)

# ==========================================
# TAB OVERVIEW
# ==========================================
tab_overview, tab_prep, tab_model, tab_eval = st.tabs([
    "📄 Overview", "🧹 Preprocessing", "⚙️ Modeling", "📊 Evaluation"
])

with tab_overview:
    st.header("1. Dataset Overview")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File berhasil diunggah!")
        
        # Menampilkan metrik ringkas
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris", df.shape[0])
        col2.metric("Jumlah Kolom", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        
        st.write("**Sampel Data Mentah:**")
        st.dataframe(df.head())
    else:
        st.info("Silakan unggah dataset (CSV) melalui sidebar di sebelah kiri untuk memulai.")

with tab_prep:
    st.header("2. Data Preprocessing")
    st.write("Proses penghapusan kolom tidak relevan, encoding, dan penyeimbangan kelas (SMOTE) akan dieksekusi dan ditampilkan di sini.")

with tab_model:
    st.header("3. Predictive Modeling")
    st.write("Proses pelatihan model Random Forest secara real-time atau pemanggilan model `.joblib` akan dilakukan di sini saat tombol eksekusi ditekan.")

with tab_eval:
    st.header("4. Model Evaluation")
    st.write("Laporan metrik seperti Accuracy, F1-Score, dan visualisasi Confusion Matrix akan muncul di sini.")