import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Olist E-Commerce Data Dashboard",
    layout="wide"
)

SVG_DASHBOARD = """<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -5px; margin-right: 8px;"><rect width="20" height="14" x="2" y="7" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>"""
SVG_TRUCK = """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -4px; margin-right: 8px;"><path d="M14 18V6a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v11a1 1 0 0 0 1 1h2"/><path d="M15 18H9"/><path d="M19 18h2a1 1 0 0 0 1-1v-3.65a1 1 0 0 0-.22-.62l-3.48-4.35A1 1 0 0 0 17.52 8H14"/><circle cx="17" cy="18" r="2"/><circle cx="7" cy="18" r="2"/></svg>"""
SVG_TARGET = """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -4px; margin-right: 8px;"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>"""
SVG_USER = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -3px; margin-right: 6px;"><circle cx="12" cy="8" r="5"/><path d="M20 21a8 8 0 1 0-16 0"/></svg>"""
SVG_ORDERS = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -3px; margin-right: 6px;"><path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"/><path d="M3 6h18"/><path d="M16 10a4 4 0 0 1-8 0"/></svg>"""
SVG_CUSTOMERS = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -3px; margin-right: 6px;"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>"""
SVG_REVENUE = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -3px; margin-right: 6px;"><line x1="12" x2="12" y1="2" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>"""

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def create_delivery_delay_df(df):
    """
    Fungsi untuk membuat dataframe keterlambatan pengiriman pada 5 kota teratas.
    """
    if df.empty:
        return pd.DataFrame(columns=['customer_city', 'delay_days'])

    df_calc = df.copy()
    
    # Hitung selisih waktu tiba aktual dengan estimasi dalam hari
    df_calc['delivery_diff_days'] = (
        df_calc['order_delivered_customer_date'] - df_calc['order_estimated_delivery_date']
    ).dt.days

    # Klasifikasi keterlambatan: jika selisih > 0 maka terlambat
    df_calc['is_delayed'] = df_calc['delivery_diff_days'] > 0
    df_calc['delay_days'] = df_calc['delivery_diff_days'].apply(lambda x: x if x > 0 else 0)

    # Ambil 5 kota dengan volume order terbanyak
    top_5_cities = df_calc['customer_city'].value_counts().head(5).index.tolist()
    df_top_cities = df_calc[df_calc['customer_city'].isin(top_5_cities)]

    # Hitung rata-rata keterlambatan per kota untuk paket yang terlambat
    delay_summary = (
        df_top_cities[df_top_cities['is_delayed'] == True]
        .groupby('customer_city')['delay_days']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    return delay_summary


def create_rfm_df(df):
    """
    Fungsi untuk membuat dataframe RFM dan mengklasifikasikan segmen pelanggan.
    """
    if df.empty:
        return pd.DataFrame(columns=['customer_unique_id', 'recency', 'frequency', 'monetary', 'Customer_Segment'])

    # Tanggal referensi: 1 hari setelah pesanan terakhir pada data terpilih
    recent_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

    rfm_df = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (recent_date - x.max()).days,
        'order_id': 'count',
        'payment_value': 'sum'
    }).reset_index()

    rfm_df.columns = ['customer_unique_id', 'recency', 'frequency', 'monetary']

    # Penilaian skor quartil RFM
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)

    try:
        rfm_df['R'] = pd.qcut(rfm_df['recency'], q=4, labels=r_labels, duplicates='drop')
        rfm_df['F'] = pd.qcut(rfm_df['frequency'].rank(method='first'), q=4, labels=f_labels, duplicates='drop')
        rfm_df['M'] = pd.qcut(rfm_df['monetary'], q=4, labels=m_labels, duplicates='drop')
    except Exception:
        rfm_df['R'], rfm_df['F'], rfm_df['M'] = 1, 1, 1

    rfm_df['RFM_Score'] = rfm_df['R'].astype(str) + rfm_df['F'].astype(str) + rfm_df['M'].astype(str)

    def segment_customer(score):
        if score >= '433':
            return 'Champion (VIP)'
        elif score >= '311':
            return 'Loyal/Active'
        elif score >= '211':
            return 'At Risk'
        else:
            return 'Lost/Hibernating'

    rfm_df['Customer_Segment'] = rfm_df['RFM_Score'].apply(segment_customer)
    return rfm_df


# ==========================================
# 3. LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    """
    Memuat dataset main_data.csv dan mengonversi kolom tanggal.
    """
    base_dir = pathlib.Path(__file__).parent.resolve()
    csv_file = base_dir / "main_data.csv"
    if not csv_file.exists():
        csv_file = pathlib.Path("main_data.csv")

    df = pd.read_csv(csv_file)

    datetime_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    for column in datetime_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])

    return df

all_df = load_data()


# ==========================================
# 4. SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown(f"### {SVG_DASHBOARD} Olist Dashboard", unsafe_allow_html=True)
    st.markdown(f"{SVG_USER} Analis: **Adhyaksa Daudi M. A.**", unsafe_allow_html=True)
    st.markdown("---")

    min_year = int(all_df['order_purchase_timestamp'].dt.year.min())
    max_year = int(all_df['order_purchase_timestamp'].dt.year.max())

    selected_year = st.slider(
        label="Filter Tahun Transaksi",
        min_value=min_year,
        max_value=max_year,
        value=2018
    )

    st.caption("Data disaring berdasarkan tahun transaksi pesanan.")

# Filter dataset berdasarkan tahun yang dipilih
filtered_df = all_df[all_df['order_purchase_timestamp'].dt.year == selected_year].copy()


# ==========================================
# 5. HALAMAN UTAMA
# ==========================================

st.markdown(
    f"## {SVG_DASHBOARD} Olist E-Commerce Data Dashboard",
    unsafe_allow_html=True
)
st.caption(f"Menampilkan ringkasan dan analisis performa untuk tahun transaksi {selected_year}")

# Validasi state data kosong
if filtered_df.empty:
    st.warning(f"Tidak ada data transaksi untuk tahun {selected_year}.")
    st.stop()

# Tiga Metrik Utama
col1, col2, col3 = st.columns(3)
with col1:
    total_orders = filtered_df['order_id'].nunique()
    st.markdown(f"**{SVG_ORDERS} Total Pesanan**", unsafe_allow_html=True)
    st.metric(label="Volume Transaksi", value=f"{total_orders:,}")
with col2:
    total_customers = filtered_df['customer_id'].nunique()
    st.markdown(f"**{SVG_CUSTOMERS} Total Pelanggan**", unsafe_allow_html=True)
    st.metric(label="Pelanggan Terdaftar", value=f"{total_customers:,}")
with col3:
    total_revenue = filtered_df['payment_value'].sum()
    st.markdown(f"**{SVG_REVENUE} Total Pendapatan**", unsafe_allow_html=True)
    st.metric(label="Nilai Transaksi (BRL)", value=f"R$ {total_revenue:,.2f}")

st.markdown("---")


# ==========================================
# SECTION 1: Keterlambatan Pengiriman
# ==========================================

st.markdown(
    f"### {SVG_TRUCK} Analisis Keterlambatan Pengiriman (Top 5 Kota)",
    unsafe_allow_html=True
)

delay_df = create_delivery_delay_df(filtered_df)

col_viz1, col_desc1 = st.columns((2, 1))

with col_viz1:
    if delay_df.empty or delay_df['delay_days'].sum() == 0:
        st.info("Tidak ada catatan paket terlambat pada 5 kota teratas untuk periode ini.")
    else:
        fig_delay, ax_delay = plt.subplots(figsize=(9, 4.5))
        
        bar_color = "#1E40AF"
        bars = ax_delay.barh(
            delay_df['customer_city'],
            delay_df['delay_days'],
            color=bar_color,
            edgecolor="#1E3A8A",
            linewidth=0.8,
            height=0.6
        )
        
        ax_delay.invert_yaxis()
        ax_delay.set_title(
            f"Rata-rata Keterlambatan di Top 5 Kota ({selected_year})",
            fontsize=12,
            fontweight="bold",
            pad=12
        )
        ax_delay.set_xlabel("Rata-rata Hari Terlambat", fontsize=10)
        ax_delay.set_ylabel("Kota Tujuan", fontsize=10)
        ax_delay.grid(axis='x', linestyle='--', alpha=0.5)
        ax_delay.spines['top'].set_visible(False)
        ax_delay.spines['right'].set_visible(False)

        for bar in bars:
            width = bar.get_width()
            ax_delay.text(
                width + 0.15,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f} hari",
                va='center',
                fontsize=9
            )

        plt.tight_layout()
        st.pyplot(fig_delay)

with col_desc1:
    st.info("**Ringkasan Operasional Logistik**")
    st.write(
        f"Grafik menampilkan 5 kota dengan volume transaksi tertinggi pada tahun {selected_year}. "
        "Metrik ini mengukur durasi keterlambatan saat paket gagal tiba sesuai tanggal estimasi. "
        "Informasi ini menjadi dasar evaluasi rute kurir dan estimasi pengiriman di wilayah terkait."
    )

st.markdown("---")


# ==========================================
# SECTION 2: RFM Analysis
# ==========================================

st.markdown(
    f"### {SVG_TARGET} Segmentasi Pelanggan (RFM Analysis)",
    unsafe_allow_html=True
)

rfm_data = create_rfm_df(filtered_df)
segment_counts = rfm_data['Customer_Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']

col_viz2, col_desc2 = st.columns((2, 1))

with col_viz2:
    if segment_counts.empty:
        st.info("Data RFM belum tersedia.")
    else:
        fig_rfm, ax_rfm = plt.subplots(figsize=(7, 5))
        
        segment_colors = {
            'Champion (VIP)': '#1D4ED8',
            'Loyal/Active': '#0D9488',
            'At Risk': '#D97706',
            'Lost/Hibernating': '#64748B'
        }
        ordered_colors = [
            segment_colors.get(seg, '#94A3B8') for seg in segment_counts['Segment']
        ]

        wedges, texts, autotexts = ax_rfm.pie(
            segment_counts['Count'],
            labels=segment_counts['Segment'],
            autopct='%1.1f%%',
            startangle=90,
            colors=ordered_colors,
            wedgeprops=dict(width=0.42, edgecolor='white', linewidth=1.5),
            pctdistance=0.78
        )

        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('white')
            autotext.set_weight('bold')

        ax_rfm.set_title(
            f"Proporsi Segmen Pelanggan ({selected_year})",
            fontsize=12,
            fontweight="bold",
            pad=15
        )

        plt.tight_layout()
        st.pyplot(fig_rfm)

with col_desc2:
    st.success("**Rekomendasi Strategi Retensi**")
    st.write(
        "1. **Champion (VIP):** Berikan penawaran eksklusif, apresiasi loyalitas, atau akses awal ke produk unggulan.\n"
        "2. **Loyal/Active:** Jaga keterlibatan dengan program rewards berkala dan rekomendasi terpersonalisasi.\n"
        "3. **At Risk:** Jalankan kampanye pengaktifan kembali melalui voucher insentif pembelian ulang.\n"
        "4. **Lost/Hibernating:** Kirim survei kepuasan singkat untuk memetakan penyebab dormansi akun."
    )

# Footer
st.caption("Dicoding Submission: Proyek Analisis Data | Adhyaksa Daudi M. A.")