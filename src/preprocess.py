"""
Skrip Prapemrosesan Data Numerik (Time-Series)
Fokus pada Missing Values, Deduplikasi, dan Format Waktu.
"""

import pandas as pd
import os
import logging
import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def find_latest_raw_file(raw_dir: str) -> str:
    """
    Mencari file CSV mentah yang paling baru (terakhir diunduh)
    di dalam folder data/raw/.
    """
    search_pattern = os.path.join(raw_dir, "raw_stock_data_*.csv")
    list_of_files = glob.glob(search_pattern)
    
    if not list_of_files:
        return ""
    
    # Mengembalikan file dengan waktu modifikasi terbaru
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def clean_time_series_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline pembersihan data harga saham:
    1. Deduplikasi
    2. Forward Fill untuk mengisi nilai harga yang kosong
    """
    logging.info(f"Memulai pembersihan data: {len(df)} baris ditemukan.")
    
    # Langkah 1: Deduplikasi berdasarkan baris yang identik
    df_cleaned = df.drop_duplicates()
    logging.info(f"Setelah deduplikasi: {len(df_cleaned)} baris tersisa.")
    
    # Langkah 2: Menangani nilai kosong (Missing Values)
    # Memakai metode 'ffill' (forward fill) untuk meneruskan harga penutupan terakhir
    df_cleaned = df_cleaned.ffill()
    
    # Jika masih ada baris teratas yang kosong (tidak bisa di ffill), hapus (dropna)
    df_cleaned = df_cleaned.dropna()
    
    logging.info("Pembersihan nilai kosong (Missing Values) selesai.")
    return df_cleaned

if __name__ == "__main__":
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    
    # 1. Identifikasi file terbaru
    target_file = find_latest_raw_file(RAW_DIR)
    
    if not target_file:
        logging.error(f"Tidak ditemukan file raw di {RAW_DIR}. Jalankan ingestion dulu!")
    else:
        logging.info(f"Memproses file: {target_file}")
        
        # 2. Baca Data
        raw_df = pd.read_csv(target_file, index_col=0)
        
        # 3. Eksekusi Pembersihan
        processed_df = clean_time_series_data(raw_df)
        
        # 4. Simpan Data Bersih
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        output_path = os.path.join(PROCESSED_DIR, "processed_stock_data.csv")
        processed_df.to_csv(output_path)
        logging.info(f"Data bersih berhasil disimpan ke: {output_path}")