# MLOps: Sistem Klasifikasi Risiko Saham

## 📌 Latar Belakang & Tujuan Proyek
Pasar saham memiliki volatilitas yang tinggi, sehingga pergerakan harga dapat berubah drastis dalam waktu singkat akibat perubahan kondisi makroekonomi atau sentimen pasar (fenomena *data drift*). Proyek ini bertujuan untuk membangun **Sistem Pendukung Keputusan (SPK)** berbasis *Machine Learning* untuk memprediksi tingkat risiko suatu saham ke dalam tiga kategori: **Aman, Waspada, dan Bahaya**. 

Sistem ini dirancang menggunakan prinsip **MLOps (Machine Learning Operations)** dengan mengimplementasikan arsitektur *Continual Learning*. Model akan secara otomatis mendeteksi anomali pada data *time-series* (OHLCV) terbaru dan melatih ulang dirinya sendiri untuk mempertahankan tingkat akurasi di *production*.

## 🛠️ Teknologi & Infrastruktur
- **Bahasa Pemrograman:** Python 3.10
- **Environment:** GitHub Codespaces (terisolasi via Docker Devcontainer)
- **Sumber Data:** API `yfinance` (Data OHLCV Historis)
- **Branching Strategy:** GitHub Flow (Standardisasi eksperimen kolaboratif)

## 📂 Struktur Direktori (Cookiecutter Data Science)
Repositori ini mengikuti konvensi standar industri agar mudah dinavigasi, direproduksi, dan diskalakan:

```text
MLOps-KlasifikasiRisikoSaham/
├── .devcontainer/        # Konfigurasi container Codespaces (Python 3.10 & Extensions)
├── config/               # File konfigurasi (hyperparameters, database config, pipeline)
├── data/
│   ├── processed/        # Data yang telah dibersihkan dan siap untuk modeling
│   └── raw/              # Data mentah langsung dari sumber (yfinance)
├── models/               # Artefak model hasil proses training (misal: .pkl, .joblib)
├── notebooks/            # Jupyter notebooks untuk Exploratory Data Analysis (EDA)
├── src/                  # Source code utama untuk pipeline MLOps
│   ├── data/             # Skrip untuk ekstraksi dan ingestion data
│   ├── features/         # Skrip rekayasa fitur (RSI, MACD, Volatilitas)
│   └── models/           # Skrip pelatihan, evaluasi, dan inferensi model
├── .gitignore            # Daftar file yang diabaikan oleh Git
├── LICENSE               # Lisensi MIT
├── README.md             # Dokumentasi utama proyek
└── requirements.txt      # Daftar dependensi library Python
```

## 🚀 Cara Menjalankan Codespaces
1. Buka repositori ini di GitHub.
2. Klik tombol `<> Code`, pilih tab **Codespaces**, lalu klik **Create codespace**.
3. Sistem akan otomatis menginstal Python 3.10 dan ekstensi yang dibutuhkan.