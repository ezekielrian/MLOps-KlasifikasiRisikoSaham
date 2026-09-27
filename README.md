# MLOps: Sistem Klasifikasi Risiko Saham

## 📌 Latar Belakang & Tujuan Proyek
Pasar saham memiliki volatilitas yang tinggi, sehingga pergerakan harga dapat berubah drastis dalam waktu singkat akibat perubahan kondisi makroekonomi atau sentimen pasar (fenomena *data drift*). Proyek ini bertujuan untuk membangun **Sistem Pendukung Keputusan (SPK)** berbasis *Machine Learning* untuk memprediksi tingkat risiko suatu saham ke dalam tiga kategori: **Aman, Waspada, dan Bahaya**. 

Sistem ini dirancang menggunakan prinsip **MLOps (Machine Learning Operations)** dengan mengimplementasikan arsitektur *Continual Learning*. Model akan secara otomatis mendeteksi anomali pada data *time-series* (OHLCV) terbaru dan melatih ulang dirinya sendiri untuk mempertahankan tingkat akurasi di *production*.

## 🛠️ Teknologi & Infrastruktur
* **Bahasa Pemrograman:** Python 3.10
* **Environment:** GitHub Codespaces (terisolasi via Docker Devcontainer)
* **Sumber Data:** API `yfinance` (Data OHLCV Historis)
* **Branching Strategy:** GitHub Flow (Standardisasi eksperimen kolaboratif)

## 📂 Struktur Direktori (Cookiecutter Data Science)
Repositori ini mengikuti konvensi standar industri agar mudah dinavigasi, direproduksi, dan diskalakan:

```text
MLOps-KlasifikasiRisikoSaham/
├── .devcontainer/        # Konfigurasi container Codespaces (Python 3.10 & Extensions)
├── configs/              # File konfigurasi (hyperparameters, database config, pipeline)
├── data/
│   ├── processed/        # Data yang telah dibersihkan dan siap untuk modeling
│   └── raw/              # Data mentah langsung dari sumber (yfinance)
├── docs/                 # Dokumentasi tambahan proyek
├── models/               # Artefak model hasil proses training (misal: .pkl, .joblib)
├── notebooks/            # Jupyter notebooks untuk Exploratory Data Analysis (EDA)
├── src/                  # Source code utama untuk pipeline MLOps
│   ├── ingest_data.py    # Skrip pengumpul data dinamis (LK-04)
│   ├── preprocess.py     # Skrip pembersihan data time-series (LK-04)
│   ├── features/         # Skrip rekayasa fitur (RSI, MACD, Volatilitas)
│   ├── models/           # Skrip pelatihan, evaluasi, dan inferensi model
│   └── hello.py          # Skrip environment testing (yfinance)
├── tests/                # Unit testing untuk pipeline dan model
├── .gitignore            # Daftar file yang diabaikan oleh Git
├── LICENSE               # Lisensi MIT
├── README.md             # Dokumentasi utama proyek
└── requirements.txt      # Daftar dependensi library Python
```

## 🚀 Cara Menjalankan Codespaces
1. Buka repositori ini di GitHub.
2. Klik tombol `<> Code`, pilih tab **Codespaces**, lalu klik **Create codespace**.
3. Sistem akan otomatis menginstal Python 3.10 dan ekstensi yang dibutuhkan.
4. Lakukan pengujian *environment* dengan menjalankan perintah: `python src/hello.py`

---

## ⚙️ Instruksi Eksekusi Data Pipeline (LK-04)

Bagian ini mendokumentasikan cara menjalankan pipeline pengumpulan dan pembersihan data dinamis (Ingestion & Preprocessing) untuk domain *financial time-series*.

### 1. Instalasi Dependencies
Pastikan environment sudah siap dengan menjalankan perintah berikut di terminal:
```bash
pip install -r requirements.txt
```

### 2. Cara Menjalankan Data Ingestion
Skrip ini akan mengambil data transaksi historis dari Yahoo Finance secara terprogram, memiliki proteksi *error handling*, dan akan menyimpannya secara non-destruktif (menggunakan nama file berbasis *timestamp*).
* **Perintah:** `python src/ingest_data.py`
* **Output:** Sebuah file CSV (contoh: `raw_stock_data_20260927_180053.csv`) berisi > 500 baris data OHLCV di dalam direktori `data/raw/`.

### 3. Cara Menjalankan Data Preprocessing
Skrip ini dirancang untuk membaca file data mentah *terbaru*, melakukan deduplikasi baris, dan *missing value handling* (Forward Fill) untuk mencegah anomali pada pemodelan.
* **Perintah:** `python src/preprocess.py`
* **Output:** Menghasilkan file data bersih dengan nama `processed_stock_data.csv` yang tersimpan di dalam direktori `data/processed/`.