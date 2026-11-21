# MLOps: Sistem Klasifikasi Risiko Saham

## 📌 Latar Belakang & Tujuan Proyek
Pasar saham memiliki volatilitas yang tinggi, sehingga pergerakan harga dapat berubah drastis dalam waktu singkat akibat perubahan kondisi makroekonomi atau sentimen pasar (fenomena *data drift*). Proyek ini bertujuan untuk membangun **Sistem Pendukung Keputusan (SPK)** berbasis *Machine Learning* untuk memprediksi tingkat risiko suatu saham ke dalam tiga kategori: **Aman, Waspada, dan Bahaya**.

Sistem ini dirancang menggunakan prinsip **MLOps (Machine Learning Operations)** dengan mengimplementasikan arsitektur *Continual Learning*. Model akan secara otomatis mendeteksi anomali pada data *time-series* (OHLCV) terbaru dan melatih ulang dirinya sendiri untuk mempertahankan tingkat akurasi di *production*.

## 🛠️ Teknologi & Infrastruktur
* **Bahasa Pemrograman:** Python 3.10
* **Environment:** GitHub Codespaces (terisolasi via Docker Devcontainer)
* **Sumber Data:** API `yfinance` (Data OHLCV Historis)
* **Data Versioning:** DVC (Data Version Control) dengan Local Remote Storage
* **Branching Strategy:** GitHub Flow (Standardisasi eksperimen kolaboratif)

## 📂 Struktur Direktori (Cookiecutter Data Science)
Repositori ini mengikuti konvensi standar industri agar mudah dinavigasi, direproduksi, dan diskalakan:

```text
MLOps-KlasifikasiRisikoSaham/
├── .dvc/                              # Konfigurasi internal Data Version Control (DVC)
├── .github/
│   ├── workflows/
│   │   └── mlflow-automation.yaml      # Workflow CI/CD GitHub Actions untuk otomatisasi pipeline dan MLflow
│   └── dependabot.yml                 # Konfigurasi pembaruan dependensi otomatis
├── configs/                           # File konfigurasi (hyperparameters, database config, pipeline)
├── data/
│   ├── processed/                     # Data yang telah dibersihkan dan siap untuk modeling
│   │   └── processed_stock_data.csv.dvc  # File tracker DVC untuk silsilah data bersih
│   └── raw/                           # Data mentah langsung dari sumber (yfinance)
│       └── raw_stock_data_*.csv.dvc   # File tracker DVC untuk silsilah data mentah
├── docs/                              # Dokumentasi tambahan proyek
├── models/                            # Artefak model hasil proses training (misal: .pkl, .joblib)
├── notebooks/                         # Jupyter notebooks untuk Exploratory Data Analysis (EDA)
├── src/                               # Source code utama untuk pipeline MLOps
│   ├── evaluate.py                    # Skrip evaluasi dan registrasi model ke MLflow
│   ├── ingest_data.py                 # Skrip pengumpul data dinamis
│   ├── preprocess.py                  # Skrip pembersihan data time-series
│   ├── features/                      # Skrip rekayasa fitur (RSI, MACD, Volatilitas)
│   ├── models/                        # Skrip pelatihan, evaluasi, dan inferensi model
│   │   └── train.py                   # Skrip pelatihan model dan pelacakan MLflow
│   └── hello.py                       # Skrip environment testing (yfinance)
├── tests/                             # Unit testing untuk pipeline dan model
├── .dvcignore                         # Daftar file yang diabaikan oleh DVC
├── .gitignore                         # Daftar file yang diabaikan oleh Git
├── app.py                             # Aplikasi Flask untuk inferensi model
├── docker-compose.yaml                # Konfigurasi orkestrasi layanan dengan Docker Compose
├── Dockerfile                         # Instruksi build image Docker
├── LICENSE                            # Lisensi MIT
├── README.md                          # Dokumentasi utama proyek
└── requirements.txt                   # Daftar dependensi library Python
```

## 🚀 Cara Menjalankan Codespaces
1. Buka repositori ini di GitHub.
2. Klik tombol `<> Code`, pilih tab **Codespaces**, lalu klik **Create codespace**.
3. Sistem akan otomatis menginstal Python 3.10 dan ekstensi yang dibutuhkan.
4. Lakukan pengujian *environment* dengan menjalankan perintah: `python src/hello.py`

---

## ⚙️ Instruksi Eksekusi Data Pipeline

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

### 🗄️ Data Lineage Management dengan DVC
Proyek ini mengimplementasikan **Data Version Control** untuk memisahkan manajemen kode dan file biner berukuran besar. Dataset aslinya tidak lagi di commit ke Git, melainkan dikirim ke *Remote Storage* menggunakan DVC, sementara Git hanya menyimpan file *pointer* berukuran ringan (file `.dvc`).

Implementasi *versioning* data ini sangat krusial karena akurasi model *machine learning* sangat bergantung pada dataset spesifik yang digunakan saat pelatihan. Mengingat data harian proyek ini diperbarui secara dinamis melalui arsitektur *continual learning*, DVC mengunci setiap transisi versi menggunakan *hash* unik agar kita selalu bisa mereproduksi versi data eksak dari masa lalu. Selain itu, pertumbuhan dataset dapat diaudit dengan jelas (*data lineage*) melalui riwayat *commit* Git, sekaligus menjaga efisiensi repositori karena file CSV berukuran besar tidak akan menyumbat kapasitas GitHub maupun memperlambat proses *clone*.

### Cara Mereproduksi / Mengunduh Data
Jika Anda baru saja melakukan *clone* repositori ini, file CSV asli belum tersedia di direktori lokal. Anda harus menariknya dari *remote storage* DVC dengan menjalankan perintah:
```bash
dvc pull
```
---

## 🤖 Pelatihan Model & MLflow Tracking
Proyek ini menggunakan **MLflow** untuk melacak eksperimen, merekam *hyperparameter*, dan membandingkan metrik evaluasi model (seperti *Akurasi* dan *F1-Score*) secara otomatis.
- **Perintah Pelatihan:** `python src/models/train.py`
- **Akses Dashboard MLflow:** Jalankan perintah `mlflow ui` di terminal, lalu buka *port* 5000 yang diteruskan oleh Codespaces.

---

## 🐳 Deployment (Docker & Flask API)
Model terbaik yang telah didaftarkan ke MLflow Model Registry kemudian dikemas ke dalam *container* terisolasi menggunakan **Docker** dan diekspos sebagai layanan REST API menggunakan **Flask**.

### Status Model Aktif (Production)
Saat ini, model yang digunakan untuk inferensi API melalui Docker adalah **Model_Risiko_Saham_Ezekiel (Versi 1)** yang berada pada stage **Production** di MLflow.

**Alasan Pemilihan:**
Model ini dipilih sebagai versi aktif karena memiliki performa klasifikasi (Akurasi/F1-Score) yang paling tinggi dan stabil selama fase eksperimen, serta telah terbukti berhasil melewati uji coba *containerization* dan validasi inferensi API tanpa kendala pada lingkungan *production*.

### Cara Menjalankan API di Docker
1. **Build Image Docker:**
   ```bash
   docker build -t model-saham-ezekiel:v1 .
   ```

2. **Jalankan Container:**
   ```bash
   docker run -p 5000:5000 -d model-saham-ezekiel:v1
   ```

3. **Uji Coba Prediksi**
   ```bash
   curl -X POST http://127.0.0.1:5000/predict \
   -H "Content-Type: application/json" \
   -d '{"features": [{"Open": 150.0, "High": 155.0, "Low": 149.0, "Volume": 10000}]}'
   ```

---

## 🧩 Orkestrasi Multi-Kontainer (Docker Compose)
Proyek ini menggunakan **Docker Compose** untuk menjalankan PostgreSQL, MLflow, dan Flask API dalam satu perintah. Layanan inferensi memuat model **`Model_Risiko_Saham_Ezekiel`** dari stage **`Staging`** di MLflow Model Registry.

### Arsitektur Layanan
* **`postgres`** menggunakan image `postgres:15` dan port internal `5432` untuk menyimpan metadata MLflow.
* **`mlflow-server`** berfungsi sebagai **MLflow Tracking & Artifact Server** dengan pemetaan port `5000:5000`. Dashboard MLflow dapat dibuka di `http://localhost:5000`.
* **`api-service`** menyediakan **Flask REST API** untuk inferensi model dengan pemetaan port `8080:5000`. API dapat diakses dari host melalui `http://localhost:8080`.

Ketiga layanan terhubung melalui jaringan *bridge* khusus bernama `ml-net`. MLflow mengakses PostgreSQL melalui `postgres:5432`, sedangkan API mengakses MLflow melalui `http://mlflow-server:5000`.

*Named volume* `pg-data` menyimpan data PostgreSQL, sedangkan `mlflow-data` menyimpan artefak MLflow. Urutan startup diatur dengan `depends_on` dan `healthcheck` agar MLflow menunggu PostgreSQL siap dan API menunggu MLflow siap.

### 1. Menjalankan Layanan
Jalankan perintah berikut dari direktori utama proyek yang memuat file `docker-compose.yaml`. Jika kontainer dari panduan deployment sebelumnya masih memakai port host `5000`, hentikan kontainer tersebut terlebih dahulu.

```bash
docker compose up -d --build
```

Cek status ketiga kontainer dengan perintah berikut.

```bash
docker compose ps
```

### 2. Pelatihan & Registrasi Model
Pastikan data hasil *preprocessing* tersedia di dalam kontainer `api-service`. Jalankan pelatihan, lalu evaluasi model untuk mendaftarkan dan mempromosikan versi yang memenuhi ambang akurasi ke stage `Staging`.

```bash
docker compose exec api-service python src/models/train.py
docker compose exec api-service python src/evaluate.py
```

### 3. Uji Prediksi & Cek Kesehatan API
Kirim permintaan `POST /predict` dengan daftar data pada kunci `features`. Jika model belum termuat di memori, `app.py` otomatis memanggil `load_model()` saat permintaan prediksi diterima.

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [{"Open": 9500, "High": 9650, "Low": 9450, "Volume": 50000000}]}'
```

Setelah prediksi berhasil, cek `GET /health` dari host untuk melihat status model yang sudah termuat.

```bash
curl http://localhost:8080/health
```

### 4. Menghentikan Layanan
Jalankan perintah berikut untuk menghentikan layanan dan mempertahankan data di volume `pg-data` dan `mlflow-data`.

```bash
docker compose down
```