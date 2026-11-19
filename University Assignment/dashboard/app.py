"""
Higher Education Data Mining Testbed
Alur CRISP-DM: Overview -> EDA -> Preprocessing -> Modeling -> Evaluation
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Higher Education Data Mining Testbed",
    page_icon="🎓",
    layout="wide",
)

# =============================================================================
# PATH KONSTAN
# =============================================================================
_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_DIR, "..", "Higher_Education_Teaching_Quality_Dataset.csv")
# Model ada di Learn Data Science/Personal Experiment/Klasifikasi/ (2 level di atas dashboard/)
MODEL_PATH = os.path.join(_DIR, "..", "..", "Personal Experiment", "Klasifikasi", "model_kualitas_pengajaran_rf.joblib")
ENCODER_PATH = os.path.join(_DIR, "..", "..", "Personal Experiment", "Klasifikasi", "label_encoder_pengajaran.joblib")

# Fallback: cari juga di folder dashboard sendiri
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(_DIR, "model_kualitas_pengajaran_rf.joblib")
if not os.path.exists(ENCODER_PATH):
    ENCODER_PATH = os.path.join(_DIR, "label_encoder_pengajaran.joblib")

TARGET_COL = "teaching_quality_level"
TARGET_ORDER = ["Poor", "Average", "Good", "Excellent"]

# Kolom yang dihapus untuk mencegah data leakage
LEAKAGE_COLS = ["record_id", "institution_id", "course_id", "faculty_id", "overall_teaching_index"]

# Fitur utama untuk simulator prediksi (berdasarkan domain knowledge & feature importance notebook)
SIMULATOR_FEATURES = [
    "student_engagement_score",
    "assignment_average_score",
    "final_exam_score",
    "quiz_average_score",
    "attendance_rate",
    "teaching_material_quality",
    "lecture_delivery_score",
    "interactive_teaching_score",
    "student_satisfaction_score",
    "instructional_effectiveness_score",
]

# =============================================================================
# HELPER: LOAD DATA
# =============================================================================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_raw_df(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Kembalikan (df, source_label). Fallback ke default CSV jika tidak ada upload."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df, "upload"
    if os.path.exists(DEFAULT_CSV):
        df = load_csv(DEFAULT_CSV)
        return df, "default"
    return None, "none"


# =============================================================================
# HELPER: PREPROCESSING
# =============================================================================
def preprocess(
    df: pd.DataFrame,
    prevent_leakage: bool,
    scaler_choice: str,
    test_size: float,
    use_smote: bool,
) -> dict:
    """
    Jalankan seluruh pipeline preprocessing.
    Kembalikan dict berisi semua artefak yang dibutuhkan tab berikutnya.
    """
    df_work = df.copy()

    # 1. Hapus kolom leakage
    dropped_cols = []
    if prevent_leakage:
        existing_drop = [c for c in LEAKAGE_COLS if c in df_work.columns]
        df_work.drop(columns=existing_drop, inplace=True)
        dropped_cols = existing_drop

    # 2. Hapus duplikat & baris dengan missing value
    df_work.dropna(inplace=True)
    df_work.drop_duplicates(inplace=True)

    # 3. Encode target dengan LabelEncoder
    le = LabelEncoder()
    le.fit(TARGET_ORDER)  # pastikan urutan konsisten: Average=0, Excellent=1, Good=2, Poor=3
    y = le.transform(df_work[TARGET_COL])

    # 4. One-Hot Encode fitur kategorikal (kecuali target)
    df_feat = df_work.drop(columns=[TARGET_COL])
    categorical_cols = df_feat.select_dtypes(include=["object", "str"]).columns.tolist()
    df_encoded = pd.get_dummies(df_feat, columns=categorical_cols, drop_first=True)

    X = df_encoded.copy()
    feature_names = X.columns.tolist()

    # 5. Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # 6. Scaling
    scaler_obj = None
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    if scaler_choice == "StandardScaler":
        scaler_obj = StandardScaler()
        X_train[numerical_cols] = scaler_obj.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler_obj.transform(X_test[numerical_cols])
    elif scaler_choice == "MinMaxScaler":
        scaler_obj = MinMaxScaler()
        X_train[numerical_cols] = scaler_obj.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler_obj.transform(X_test[numerical_cols])

    # 7. SMOTE
    y_train_before = pd.Series(y_train).value_counts().sort_index()
    X_train_final, y_train_final = X_train, y_train
    y_train_after = None
    if use_smote:
        sm = SMOTE(random_state=42)
        X_train_final, y_train_final = sm.fit_resample(X_train, y_train)
        y_train_after = pd.Series(y_train_final).value_counts().sort_index()

    return {
        "X_train": X_train_final,
        "X_test": X_test,
        "y_train": y_train_final,
        "y_test": y_test,
        "label_encoder": le,
        "feature_names": feature_names,
        "dropped_cols": dropped_cols,
        "categorical_cols": categorical_cols,
        "scaler_choice": scaler_choice,
        "scaler_obj": scaler_obj,          # None jika scaler_choice=="None"
        "scaler_num_cols": list(numerical_cols),  # kolom numerik yang di-scale
        "y_train_before_smote": y_train_before,
        "y_train_after_smote": y_train_after,
        "X_train_raw_size": len(X_train),
        "X_test_raw_size": len(X_test),
        "df_original": df_work,
    }


# =============================================================================
# HELPER: TRAINING
# =============================================================================
def train_rf(
    X_train, y_train,
    n_estimators: int,
    max_depth: int,
    max_features: str,
    cv_folds: int,
) -> tuple:
    """Latih Random Forest dan hitung CV F1-weighted. Kembalikan (model, cv_score)."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
    model.fit(X_train, y_train)
    return model, float(np.mean(cv_scores))


# =============================================================================
# HELPER: EVALUATION PLOTS
# =============================================================================
def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Aktual", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 10) -> plt.Figure:
    importances = model.feature_importances_
    imp_df = (
        pd.DataFrame({"Fitur": feature_names, "Kepentingan": importances})
        .sort_values("Kepentingan", ascending=True)
        .tail(top_n)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("crest", len(imp_df))
    ax.barh(imp_df["Fitur"], imp_df["Kepentingan"], color=colors)
    ax.set_xlabel("Skor Kepentingan (Feature Importance)", fontsize=10)
    ax.set_title(f"Top {top_n} Fitur Paling Berpengaruh", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    counts = df[TARGET_COL].value_counts().reindex(TARGET_ORDER, fill_value=0)
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = ["#D94F3D", "#E8A838", "#5B8DB8", "#4CAF7A"]
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 60,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax.set_xlabel("Kelas Kualitas Pengajaran", fontsize=11)
    ax.set_ylabel("Jumlah Sampel", fontsize=11)
    ax.set_title("Distribusi Kelas Target: teaching_quality_level", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    return fig


def plot_smote_comparison(before: pd.Series, after: pd.Series, le: LabelEncoder) -> plt.Figure:
    class_names = le.inverse_transform(sorted(before.index))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    colors = ["#5B8DB8", "#4CAF7A", "#E8A838", "#D94F3D"]
    for ax, data, title in zip(
        axes,
        [before, after],
        ["Sebelum SMOTE", "Sesudah SMOTE"],
    ):
        vals = [data.get(i, 0) for i in sorted(before.index)]
        ax.bar(class_names, vals, color=colors[: len(class_names)], edgecolor="white")
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, str(v), ha="center", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Kelas")
        ax.set_ylabel("Jumlah Sampel")
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("Perbandingan Distribusi Kelas: Sebelum vs Sesudah SMOTE", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_boxplot_by_class(df: pd.DataFrame, feature: str) -> plt.Figure:
    order = [c for c in TARGET_ORDER if c in df[TARGET_COL].unique()]
    palette = {"Poor": "#D94F3D", "Average": "#E8A838", "Good": "#5B8DB8", "Excellent": "#4CAF7A"}
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df, x=TARGET_COL, y=feature, order=order,
        palette=palette, linewidth=1.2, ax=ax,
    )
    ax.set_title(f"Distribusi {feature} per Kelas Kualitas Pengajaran", fontsize=12, fontweight="bold")
    ax.set_xlabel("Kelas Kualitas Pengajaran")
    ax.set_ylabel(feature)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Pilih kolom pedagogis/akademis utama yang ada di dataset
    priority = [
        "student_engagement_score", "assignment_average_score", "final_exam_score",
        "quiz_average_score", "attendance_rate", "teaching_material_quality",
        "lecture_delivery_score", "interactive_teaching_score", "student_satisfaction_score",
        "instructional_effectiveness_score", "learning_outcome_achievement",
        "assessment_fairness_score", "practical_session_score",
    ]
    cols_plot = [c for c in priority if c in num_cols][:13]
    corr = df[cols_plot].corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7},
        vmin=-1, vmax=1,
    )
    ax.set_title("Heatmap Korelasi Fitur Pedagogis & Akademik Utama", fontsize=12, fontweight="bold", pad=10)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    return fig


# =============================================================================
# INISIALISASI SESSION STATE
# =============================================================================
for key in ["model", "label_encoder", "feature_names", "prep_result", "cv_score", "model_source"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =============================================================================
# HEADER
# =============================================================================
st.title("🎓 Higher Education Data Mining Testbed")
st.caption("Pipeline CRISP-DM end-to-end: Pemahaman Data, EDA, Preprocessing, Pemodelan, Evaluasi")

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Testbed Configuration")

    # -- Dataset Input --
    st.subheader("📂 Dataset Input")
    uploaded_file = st.file_uploader("Upload Dataset CSV (Opsional)", type=["csv"])

    st.divider()

    # -- Data Leakage & Feature Preparation --
    st.subheader("🛡️ Data Leakage & Feature Preparation")
    prevent_leakage = st.checkbox(
        "Cegah Data Leakage",
        value=True,
        help=f"Hapus kolom: {', '.join(LEAKAGE_COLS)}",
    )
    scaler_choice = st.selectbox("Feature Scaler", ["None", "StandardScaler", "MinMaxScaler"], index=1)
    test_size = st.slider("Test Size (Holdout)", 0.10, 0.40, 0.20, step=0.05, format="%.2f")
    use_smote = st.toggle("Aktifkan SMOTE (Class Balancing)", value=True)

    st.divider()

    # -- Model & Hyperparameter --
    st.subheader("🧠 Model & Hyperparameter Tuning")
    cv_folds = st.slider("Cross-Validation Folds (K)", 3, 10, 3)
    n_estimators = st.slider("n_estimators", 50, 200, 100, step=10)
    max_depth = st.slider("max_depth", 5, 40, 20, step=5)
    max_features = st.selectbox("max_features", ["sqrt", "log2"], index=0)

# =============================================================================
# LOAD DATA
# =============================================================================
df_raw, data_source = get_raw_df(uploaded_file)

# =============================================================================
# TAB LAYOUT
# =============================================================================
tab_overview, tab_eda, tab_prep, tab_model, tab_eval = st.tabs(
    ["📄 Overview", "🔍 EDA", "🧹 Preprocessing", "⚙️ Modeling", "📊 Evaluation"]
)

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab_overview:
    st.header("Pemahaman Dataset")

    if data_source == "none":
        st.error("Dataset default tidak ditemukan dan belum ada file yang diunggah. Upload CSV melalui sidebar.")
        st.stop()

    if data_source == "default":
        st.info("Dataset default (`Higher_Education_Teaching_Quality_Dataset.csv`) dimuat secara otomatis. Upload file CSV di sidebar untuk menggantinya.")
    else:
        st.success("Dataset berhasil diunggah.")

    # 4 Metrik Utama
    total_missing = int(df_raw.isnull().sum().sum())
    n_classes = df_raw[TARGET_COL].nunique() if TARGET_COL in df_raw.columns else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Baris", f"{df_raw.shape[0]:,}")
    c2.metric("Total Kolom", df_raw.shape[1])
    c3.metric("Missing Values", total_missing)
    c4.metric("Jumlah Kelas Target", n_classes)

    st.subheader("Sampel Data Mentah (10 Baris Pertama)")
    st.dataframe(df_raw.head(10), use_container_width=True)

    with st.expander("Ringkasan Tipe Data & Statistik Deskriptif"):
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.write("**Tipe Data per Kolom**")
            dtype_df = pd.DataFrame({"Kolom": df_raw.dtypes.index, "Tipe": df_raw.dtypes.values})
            st.dataframe(dtype_df, use_container_width=True, height=400)
        with col_b:
            st.write("**Statistik Deskriptif (Kolom Numerik)**")
            st.dataframe(df_raw.describe(), use_container_width=True)

# =============================================================================
# TAB 2: EDA
# =============================================================================
with tab_eda:
    st.header("Eksplorasi Data (EDA)")

    if data_source == "none":
        st.warning("Belum ada dataset. Upload CSV di sidebar.")
    else:
        # -- Distribusi Kelas Target --
        st.subheader("Distribusi Kelas Target")
        if TARGET_COL in df_raw.columns:
            fig_dist = plot_class_distribution(df_raw)
            st.pyplot(fig_dist)
            plt.close(fig_dist)

            class_counts = df_raw[TARGET_COL].value_counts().reindex(TARGET_ORDER, fill_value=0)
            dominant = class_counts.idxmax()
            minority = class_counts.idxmin()
            imbalance_ratio = class_counts.max() / class_counts.min()
            st.info(
                f"Kelas **{dominant}** mendominasi data ({class_counts[dominant]:,} sampel). "
                f"Kelas **{minority}** adalah yang paling sedikit ({class_counts[minority]:,} sampel). "
                f"Rasio ketimpangan: **{imbalance_ratio:.1f}x** — ini alasan SMOTE dibutuhkan."
            )
        else:
            st.warning(f"Kolom `{TARGET_COL}` tidak ditemukan pada dataset.")

        st.divider()

        # -- Heatmap Korelasi --
        st.subheader("Korelasi Fitur Pedagogis & Akademik Utama")
        try:
            fig_corr = plot_correlation_heatmap(df_raw)
            st.pyplot(fig_corr)
            plt.close(fig_corr)
        except Exception as e:
            st.error(f"Gagal membuat heatmap korelasi: {e}")

        st.divider()

        # -- Analisis Interaktif Antar-Fitur --
        st.subheader("Analisis Distribusi Fitur per Kelas")
        num_cols_eda = df_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
        # Prioritaskan fitur simulator agar relevan
        priority_eda = [c for c in SIMULATOR_FEATURES if c in num_cols_eda]
        rest_eda = [c for c in num_cols_eda if c not in priority_eda]
        ordered_eda = priority_eda + rest_eda

        selected_feature = st.selectbox(
            "Pilih Fitur untuk Dieksplorasi",
            options=ordered_eda,
            index=0,
        )

        if TARGET_COL in df_raw.columns and selected_feature:
            fig_box = plot_boxplot_by_class(df_raw, selected_feature)
            st.pyplot(fig_box)
            plt.close(fig_box)

            # Insight otomatis
            try:
                medians = df_raw.groupby(TARGET_COL)[selected_feature].median().reindex(TARGET_ORDER)
                best_class = medians.idxmax()
                st.info(
                    f"Median **{selected_feature}** tertinggi ada di kelas **{best_class}** "
                    f"({medians[best_class]:.2f}). Pola ini mengindikasikan korelasi positif fitur ini "
                    "terhadap kualitas pengajaran yang lebih tinggi."
                )
            except Exception:
                pass

# =============================================================================
# TAB 3: PREPROCESSING
# =============================================================================
with tab_prep:
    st.header("Tahapan Preprocessing Data")

    if data_source == "none":
        st.warning("Belum ada dataset. Upload CSV di sidebar.")
    else:
        # Jalankan preprocessing dan simpan ke session state
        try:
            prep = preprocess(df_raw, prevent_leakage, scaler_choice, test_size, use_smote)
            st.session_state["prep_result"] = prep
        except Exception as e:
            st.error(f"Preprocessing gagal: {e}")
            st.stop()

        # -- Kolom yang Dihapus --
        st.subheader("1. Kolom yang Dihapus (Pencegahan Data Leakage)")
        if prep["dropped_cols"]:
            st.write(f"Kolom berikut dihapus karena berpotensi menyebabkan data leakage:")
            st.code(", ".join(prep["dropped_cols"]))
        else:
            st.info("Pencegahan leakage tidak diaktifkan. Semua kolom dipertahankan.")

        st.subheader("2. Encoding Fitur Kategorikal")
        if prep["categorical_cols"]:
            st.write(f"Kolom kategorikal berikut di-encode dengan **One-Hot Encoding** (`drop_first=True`):")
            st.code(", ".join(prep["categorical_cols"]))
        else:
            st.info("Tidak ada kolom kategorikal yang ditemukan (selain target).")
        st.write(f"Kolom target (`{TARGET_COL}`) di-encode dengan **LabelEncoder**: "
                f"`Average=0, Excellent=1, Good=2, Poor=3`")

        st.subheader("3. Scaling Fitur Numerik")
        if scaler_choice == "None":
            st.info("Scaling tidak diaktifkan.")
        else:
            st.write(f"Metode scaling aktif: **{scaler_choice}** (fit pada train, transform pada test).")

        st.divider()

        # -- Train/Test Split --
        st.subheader("4. Pembagian Data Train vs Test")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Sampel (setelah cleaning)", prep["X_train_raw_size"] + prep["X_test_raw_size"])
        c2.metric("Train Set", f"{prep['X_train_raw_size']:,} ({(1-test_size)*100:.0f}%)")
        c3.metric("Test Set", f"{prep['X_test_raw_size']:,} ({test_size*100:.0f}%)")

        st.divider()

        # -- SMOTE --
        st.subheader("5. Class Balancing (SMOTE)")
        if use_smote and prep["y_train_after_smote"] is not None:
            le = prep["label_encoder"]
            before = prep["y_train_before_smote"]
            after = prep["y_train_after_smote"]

            # Tabel perbandingan
            before_named = {le.inverse_transform([k])[0]: v for k, v in before.items()}
            after_named = {le.inverse_transform([k])[0]: v for k, v in after.items()}
            comp_df = pd.DataFrame({
                "Kelas": TARGET_ORDER,
                "Sebelum SMOTE": [before_named.get(c, 0) for c in TARGET_ORDER],
                "Sesudah SMOTE": [after_named.get(c, 0) for c in TARGET_ORDER],
            })
            comp_df["Penambahan"] = comp_df["Sesudah SMOTE"] - comp_df["Sebelum SMOTE"]
            st.dataframe(comp_df.set_index("Kelas"), use_container_width=True)

            fig_smote = plot_smote_comparison(before, after, le)
            st.pyplot(fig_smote)
            plt.close(fig_smote)

            st.success(
                f"SMOTE berhasil menyeimbangkan data latih. "
                f"Total sampel train naik dari **{before.sum():,}** menjadi **{after.sum():,}**."
            )
        else:
            st.info("SMOTE tidak diaktifkan. Data latih digunakan tanpa resampling.")

        st.divider()

        # -- Pratinjau X_train --
        st.subheader("6. Pratinjau X_train Setelah Preprocessing (5 Baris Pertama)")
        st.dataframe(prep["X_train"].head(), use_container_width=True)
        st.caption(f"Dimensi X_train: {prep['X_train'].shape[0]:,} baris x {prep['X_train'].shape[1]} kolom")

# =============================================================================
# TAB 4: MODELING
# =============================================================================
with tab_model:
    st.header("Pemodelan Prediktif")

    if st.session_state["prep_result"] is None:
        st.warning("Preprocessing belum dijalankan. Buka tab **Preprocessing** terlebih dahulu agar pipeline siap.")
    else:
        prep = st.session_state["prep_result"]

        # -- Bagian A: Pelatihan / Load Model --
        st.subheader("A. Pelatihan & Pemilihan Model")
        col_train, col_load = st.columns(2)

        with col_train:
            st.write(
                f"Konfigurasi: `n_estimators={n_estimators}`, `max_depth={max_depth}`, "
                f"`max_features={max_features}`, `CV={cv_folds}-Fold`"
            )
            if st.button("🚀 Latih Model Random Forest", use_container_width=True, type="primary"):
                with st.spinner("Melatih model... Proses ini mungkin memerlukan beberapa menit."):
                    try:
                        model, cv_score = train_rf(
                            prep["X_train"], prep["y_train"],
                            n_estimators, max_depth, max_features, cv_folds,
                        )
                        st.session_state["model"] = model
                        st.session_state["label_encoder"] = prep["label_encoder"]
                        st.session_state["feature_names"] = prep["feature_names"]
                        st.session_state["cv_score"] = cv_score
                        st.session_state["model_source"] = "trained"
                        st.success(f"Model berhasil dilatih! CV F1-Weighted rata-rata: **{cv_score:.4f}**")
                    except Exception as e:
                        st.error(f"Pelatihan gagal: {e}")

        with col_load:
            st.write("Muat model optimal yang sudah disimpan sebelumnya (tanpa proses training ulang).")
            model_exists = os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)
            if not model_exists:
                st.caption(f"File model tidak ditemukan di: `{MODEL_PATH}`")
            if st.button(
                "📂 Muat Model Tersimpan",
                use_container_width=True,
                disabled=not model_exists,
            ):
                try:
                    loaded_model = joblib.load(MODEL_PATH)
                    loaded_le = joblib.load(ENCODER_PATH)
                    st.session_state["model"] = loaded_model
                    st.session_state["label_encoder"] = loaded_le
                    # Feature names dari model yang dimuat = feature_importances_ index
                    st.session_state["feature_names"] = prep["feature_names"]
                    st.session_state["cv_score"] = None
                    st.session_state["model_source"] = "loaded"
                    st.success("Model tersimpan berhasil dimuat.")
                except Exception as e:
                    st.error(f"Gagal memuat model: {e}")

        # Tampilkan status model aktif
        if st.session_state["model"] is not None:
            source_label = "dilatih baru" if st.session_state["model_source"] == "trained" else "dimuat dari file"
            st.info(
                f"Model aktif: **Random Forest** ({source_label}). "
                + (f"CV F1-Weighted: **{st.session_state['cv_score']:.4f}**" if st.session_state["cv_score"] else "")
            )

        st.divider()

        # -- Bagian B: Simulator Prediksi Interaktif --
        st.subheader("B. Simulator Prediksi Kualitas Pengajaran")

        if st.session_state["model"] is None:
            st.warning("Latih atau muat model terlebih dahulu (Bagian A) untuk menggunakan simulator.")
        else:
            st.write(
                "Isi nilai fitur utama di bawah ini. Fitur lain yang tidak ditampilkan "
                "akan diisi otomatis dengan nilai **median** dari dataset."
            )

            # Hitung median/modus dari data original untuk default
            df_orig = prep["df_original"]
            feature_names = st.session_state["feature_names"]

            # Siapkan vector input default (median semua fitur numerik)
            default_values = {}
            for col in df_orig.columns:
                if col == TARGET_COL:
                    continue
                if df_orig[col].dtype in [np.float64, np.int64]:
                    default_values[col] = float(df_orig[col].median())

            with st.form("simulator_form"):
                st.write("**Input Fitur Utama:**")
                sim_cols = st.columns(2)
                input_vals = {}

                # Definisi range tiap fitur simulator
                sim_ranges = {
                    "student_engagement_score": (0.0, 100.0, 1.0),
                    "assignment_average_score": (0.0, 100.0, 1.0),
                    "final_exam_score": (0.0, 100.0, 1.0),
                    "quiz_average_score": (0.0, 100.0, 1.0),
                    "attendance_rate": (0.0, 100.0, 0.5),
                    "teaching_material_quality": (0.0, 100.0, 1.0),
                    "lecture_delivery_score": (0.0, 100.0, 1.0),
                    "interactive_teaching_score": (0.0, 100.0, 1.0),
                    "student_satisfaction_score": (0.0, 100.0, 1.0),
                    "instructional_effectiveness_score": (0.0, 100.0, 1.0),
                }

                for idx, feat in enumerate(SIMULATOR_FEATURES):
                    if feat not in df_orig.columns:
                        continue
                    min_v, max_v, step_v = sim_ranges.get(feat, (0.0, 100.0, 1.0))
                    median_v = default_values.get(feat, 50.0)
                    with sim_cols[idx % 2]:
                        input_vals[feat] = st.slider(
                            feat.replace("_", " ").title(),
                            min_value=min_v,
                            max_value=max_v,
                            value=float(np.clip(median_v, min_v, max_v)),
                            step=step_v,
                        )

                submitted = st.form_submit_button("🔮 Prediksi Kualitas Pengajaran", type="primary", use_container_width=True)

            if submitted:
                try:
                    model = st.session_state["model"]
                    le = st.session_state["label_encoder"]
                    feature_names = st.session_state["feature_names"]

                    # Bangun sample dari baris pertama df_original sebagai template,
                    # override dengan nilai input user, lalu encode + scale sesuai pipeline training.
                    sample_row = df_orig.drop(columns=[TARGET_COL]).iloc[[0]].copy()
                    for feat, val in input_vals.items():
                        if feat in sample_row.columns:
                            sample_row[feat] = val

                    # One-hot encode
                    categorical_cols = prep["categorical_cols"]
                    sample_encoded = pd.get_dummies(sample_row, columns=categorical_cols, drop_first=True)

                    # Align kolom dengan feature_names
                    for col in feature_names:
                        if col not in sample_encoded.columns:
                            sample_encoded[col] = 0
                    sample_encoded = sample_encoded[feature_names]

                    # Apply scaler jika aktif (wajib agar distribusi input cocok dengan data training)
                    scaler_obj = prep.get("scaler_obj")
                    scaler_num_cols = prep.get("scaler_num_cols", [])
                    if scaler_obj is not None and scaler_num_cols:
                        cols_to_scale = [c for c in scaler_num_cols if c in sample_encoded.columns]
                        sample_encoded[cols_to_scale] = scaler_obj.transform(
                            sample_encoded[cols_to_scale]
                        )

                    # Predict
                    pred_encoded = model.predict(sample_encoded)[0]
                    pred_proba = model.predict_proba(sample_encoded)[0]
                    pred_label = le.inverse_transform([pred_encoded])[0]
                    class_names = le.classes_

                    # Tampilkan hasil
                    color_map = {"Poor": "🔴", "Average": "🟡", "Good": "🔵", "Excellent": "🟢"}
                    icon = color_map.get(pred_label, "⚪")
                    st.markdown(f"### Hasil Prediksi: {icon} **{pred_label}**")

                    # Bar chart probabilitas
                    prob_df = pd.DataFrame({"Kelas": class_names, "Probabilitas": pred_proba}).sort_values(
                        "Probabilitas", ascending=True
                    )
                    fig_prob, ax_prob = plt.subplots(figsize=(6, 3))
                    colors_prob = ["#D94F3D", "#E8A838", "#5B8DB8", "#4CAF7A"]
                    color_dict = dict(zip(["Poor", "Average", "Good", "Excellent"], colors_prob))
                    bar_colors = [color_dict.get(c, "#888") for c in prob_df["Kelas"]]
                    ax_prob.barh(prob_df["Kelas"], prob_df["Probabilitas"], color=bar_colors)
                    for i, (_, row_p) in enumerate(prob_df.iterrows()):
                        ax_prob.text(row_p["Probabilitas"] + 0.005, i, f"{row_p['Probabilitas']:.1%}", va="center", fontsize=9)
                    ax_prob.set_xlim(0, 1.15)
                    ax_prob.set_xlabel("Probabilitas")
                    ax_prob.set_title("Probabilitas Per Kelas", fontsize=11, fontweight="bold")
                    ax_prob.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_prob)
                    plt.close(fig_prob)

                except Exception as e:
                    st.error(f"Prediksi gagal: {e}")

# =============================================================================
# TAB 5: EVALUATION
# =============================================================================
with tab_eval:
    st.header("Evaluasi Model")

    if st.session_state["model"] is None:
        st.warning(
            "Model belum tersedia. Jalankan pelatihan atau muat model di tab **Modeling** terlebih dahulu."
        )
    elif st.session_state["prep_result"] is None:
        st.warning("Data preprocessing belum tersedia. Buka tab **Preprocessing** terlebih dahulu.")
    else:
        model = st.session_state["model"]
        le = st.session_state["label_encoder"]
        feature_names = st.session_state["feature_names"]
        prep = st.session_state["prep_result"]

        X_test = prep["X_test"]
        y_test = prep["y_test"]

        # Align kolom X_test dengan feature_names model (safeguard)
        try:
            X_test_aligned = X_test[feature_names]
        except KeyError:
            # Tambah kolom yang hilang dengan nilai 0
            X_test_aligned = X_test.reindex(columns=feature_names, fill_value=0)

        try:
            y_pred = model.predict(X_test_aligned)
        except Exception as e:
            st.error(f"Prediksi pada test set gagal: {e}")
            st.stop()

        class_names = le.classes_

        # -- Metrik Utama --
        st.subheader("Metrik Utama pada Data Uji")
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc:.4f}", f"{acc*100:.2f}%")
        m2.metric("Macro F1-Score", f"{f1_macro:.4f}")
        m3.metric("Weighted F1-Score", f"{f1_weighted:.4f}")

        st.divider()

        # -- Classification Report --
        st.subheader("Classification Report per Kelas")
        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report_dict).T
        # Hapus baris agregat rata-rata agar lebih bersih, tampilkan hanya per kelas
        per_class = report_df.loc[class_names].rename(columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-Score",
            "support": "Support",
        })
        per_class["Support"] = per_class["Support"].astype(int)
        st.dataframe(per_class.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"}), use_container_width=True)

        st.divider()

        # -- Confusion Matrix --
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = plot_confusion_matrix(cm, class_names)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        st.divider()

        # -- Feature Importance --
        st.subheader("Top 10 Feature Importance")
        try:
            fig_fi = plot_feature_importance(model, feature_names, top_n=10)
            st.pyplot(fig_fi)
            plt.close(fig_fi)

            # Narasi kesimpulan
            importances = model.feature_importances_
            imp_df = pd.DataFrame({"Fitur": feature_names, "Kepentingan": importances})
            top3 = imp_df.nlargest(3, "Kepentingan")["Fitur"].tolist()
            st.info(
                f"Tiga faktor pedagogis paling berpengaruh terhadap kualitas pengajaran adalah: "
                f"**{top3[0]}**, **{top3[1]}**, dan **{top3[2]}**. "
                "Peningkatan pada fitur-fitur ini berpotensi meningkatkan kelas kualitas pengajaran secara signifikan."
            )
        except Exception as e:
            st.error(f"Gagal menampilkan feature importance: {e}")
