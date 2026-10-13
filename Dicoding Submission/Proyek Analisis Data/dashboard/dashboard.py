import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 1. Konfigurasi Halaman Dashboard
st.set_page_config(
    page_title="E-Commerce Olist Dashboard",
    page_icon="🛒",
    layout="wide"
)

# 2. Fungsi Load Data (Menggunakan decorator st.cache_data agar cepat dimuat)
@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv")
    
    # Pastikan tipe data datetime kembali dibaca dengan benar
    datetime_columns = ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
    for column in datetime_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
    return df

df = load_data()

# 3. Sidebar untuk Navigasi & Filter Waktu
with st.sidebar:
    st.image("https://raw.githubusercontent.com/dicodingacademy/assets/main/logo.png", width=200)
    st.title("Adhyaksa Daudi M. A.")
    st.markdown("---")
    
    # Filter Berdasarkan Tahun Transaksi
    min_year = df['order_purchase_timestamp'].dt.year.min()
    max_year = df['order_purchase_timestamp'].dt.year.max()
    
    selected_year = st.slider(
        label="Pilih Tahun Transaksi",
        min_value=min_year,
        max_value=max_year,
        value=2018 # Default di 2018 sesuai pertanyaan bisnis
    )

# Memfilter data utama berdasarkan tahun yang dipilih di sidebar
filtered_df = df[df['order_purchase_timestamp'].dt.year == selected_year].copy()

# 4. Bagian Header Utama
st.title("🛒 Olist E-Commerce Data Dashboard")
st.markdown(f"**Menampilkan Data Tahun: {selected_year}**")

# Membuat 3 Tab Metrik Utama
col1, col2, col3 = st.columns(3)
with col1:
    total_orders = filtered_df['order_id'].nunique()
    st.metric("Total Pesanan", value=f"{total_orders:,}")
with col2:
    total_customers = filtered_df['customer_id'].nunique()
    st.metric("Total Pelanggan", value=f"{total_customers:,}")
with col3:
    total_revenue = filtered_df['payment_value'].sum()
    # Menggunakan kurs BRL (Brazilian Real) karena data dari Brazil, atau USD
    st.metric("Total Pendapatan", value=f"R$ {total_revenue:,.2f}")

st.markdown("---")

# 5. Visualisasi 1: Analisis Keterlambatan Pengiriman (Kiri) & Info (Kanan)
st.subheader("🚚 Analisis Keterlambatan Pengiriman (Top 5 Kota)")

col_viz1, col_desc1 = st.columns((2, 1))

with col_viz1:
    # Logic perhitungan keterlambatan
    filtered_df['delivery_diff_days'] = (filtered_df['order_delivered_customer_date'] - filtered_df['order_estimated_delivery_date']).dt.days
    filtered_df['is_delayed'] = filtered_df['delivery_diff_days'] > 0
    filtered_df['delay_days'] = filtered_df['delivery_diff_days'].apply(lambda x: x if x > 0 else 0)
    
    # Mencari top 5 kota di tahun terpilih
    top_5_cities = filtered_df['customer_city'].value_counts().head(5).index.tolist()
    df_top_cities = filtered_df[filtered_df['customer_city'].isin(top_5_cities)]
    
    # Menghitung rata-rata keterlambatan
    delay_summary = df_top_cities[df_top_cities['is_delayed'] == True].groupby('customer_city')['delay_days'].mean().sort_values(ascending=False).reset_index()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='delay_days', y='customer_city', data=delay_summary, palette='viridis', ax=ax)
    ax.set_title(f'Rata-rata Keterlambatan di Top 5 Kota ({selected_year})', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Rata-rata Hari Terlambat', fontsize=12)
    ax.set_ylabel('Kota Tujuan', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    for i, val in enumerate(delay_summary['delay_days']):
        ax.text(val + 0.1, i, f"{val:.1f} hari", va='center', fontsize=11)
    
    # Tampilkan plot di Streamlit
    st.pyplot(fig)

with col_desc1:
    st.info("**Insight Bisnis:**")
    st.write(f"Grafik di samping menampilkan daftar 5 kota dengan volume pesanan terbanyak pada tahun {selected_year}. Di antara kota-kota tersebut, kita dapat melihat kota mana yang mencatat rata-rata hari keterlambatan paling lama saat paket gagal tiba sesuai estimasi.")
    
st.markdown("---")

# 6. Visualisasi 2: Segmentasi RFM (Tengah)
st.subheader("🎯 Segmentasi Pelanggan VIP (RFM Analysis)")

# Logic RFM (Kita hitung ulang khusus untuk pelanggan di tahun terpilih agar dinamis)
recent_date = filtered_df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
rfm_df = filtered_df.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (recent_date - x.max()).days,
    'order_id': 'count',
    'payment_value': 'sum'
}).reset_index()
rfm_df.columns = ['customer_unique_id', 'recency', 'frequency', 'monetary']

# Fungsi Scoring RFM (Disederhanakan untuk dashboard)
def get_rfm_segments(df):
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)
    
    try:
        df['R'] = pd.qcut(df['recency'], q=4, labels=r_labels, duplicates='drop')
        df['F'] = pd.qcut(df['frequency'].rank(method='first'), q=4, labels=f_labels, duplicates='drop')
        df['M'] = pd.qcut(df['monetary'], q=4, labels=m_labels, duplicates='drop')
    except:
        # Fallback jika data terlalu sedikit untuk di-qcut
        df['R'], df['F'], df['M'] = 1, 1, 1
        
    df['RFM_Score'] = df['R'].astype(str) + df['F'].astype(str) + df['M'].astype(str)
    
    def segment_customer(score):
        if score >= '433': return 'Champion (VIP)'
        elif score >= '311': return 'Loyal/Active'
        elif score >= '211': return 'At Risk'
        else: return 'Lost/Hibernating'
        
    df['Customer_Segment'] = df['RFM_Score'].apply(segment_customer)
    return df

rfm_segmented = get_rfm_segments(rfm_df)
segment_counts = rfm_segmented['Customer_Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']

# Layout untuk RFM
col_viz2, col_desc2 = st.columns((2, 1))

with col_viz2:
    # Plotting Pie Chart
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    ax2.pie(segment_counts['Count'], labels=segment_counts['Segment'], autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.05, 0, 0, 0))
    ax2.set_title(f'Proporsi Segmen Pelanggan ({selected_year})', fontsize=14, fontweight='bold', pad=20)
    
    st.pyplot(fig2)

with col_desc2:
    st.success("**Rekomendasi Aksi:**")
    st.write("1. **Champion (VIP):** Berikan akses eksklusif ke produk baru atau program *Loyalty Points*.")
    st.write("2. **At Risk:** Kirimkan email promosi *re-engagement* diskon untuk memancing transaksi ulang.")
    st.write("3. **Lost:** Lakukan survei singkat mengapa mereka berhenti berbelanja.")

st.caption("Copyright (c) Adhyaksa Daudi M. A. - Dicoding Academy 2026")