# E-Commerce Data Analysis Dashboard ✨
Dashboard ini merupakan tugas akhir (submission) untuk kelas Belajar Analisis Data dengan Python di Dicoding. Proyek ini menganalisis dataset *E-Commerce* publik dengan fokus pada dua pertanyaan bisnis utama:
1. Analisis performa pengiriman logistik (tingkat keterlambatan) di 5 kota teratas.
2. Segmentasi pelanggan menggunakan metode *RFM (Recency, Frequency, Monetary) Analysis*.

## Setup Environment - Anaconda
Jika menggunakan Anaconda, jalankan perintah berikut di terminal/command prompt:
```bash
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal
Jika menggunakan Shell/Terminal standar dengan `pipenv`, jalankan perintah berikut:

```bash
mkdir proyek_analisis_data
cd proyek_analisis_data
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Run Streamlit App
Setelah *environment* siap dan *library* terinstal, jalankan perintah berikut untuk membuka *dashboard*:

```bash
cd dashboard
streamlit run dashboard.py
```