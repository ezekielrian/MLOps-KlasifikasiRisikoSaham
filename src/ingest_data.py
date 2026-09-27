"""
Skrip Ingestion Data Dinamis (Yahoo Finance)
"""

import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime

# Konfigurasi logging untuk mencatat proses eksekusi
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_stock_data(ticker_list: list, fetch_period: str) -> pd.DataFrame:
    """
    Mengambil data pasar saham secara dinamis menggunakan yfinance.
    Dilengkapi dengan penanganan error koneksi (try-except).
    """
    logging.info(f"Mulai mengambil data untuk: {ticker_list}")
    try:
        # Mengunduh data historis
        raw_data = yf.download(ticker_list, period=fetch_period)
        
        if raw_data.empty:
            logging.warning("Data berhasil ditarik, namun kosong.")
            return pd.DataFrame()
            
        logging.info("Data berhasil diambil dari sumber eksternal.")
        return raw_data
        
    except Exception as e:
        # Menangani error koneksi atau kegagalan API
        logging.error(f"Gagal mengambil data dari Yahoo Finance: {e}")
        return pd.DataFrame()

def save_raw_data(data: pd.DataFrame, output_dir: str) -> None:
    """
    Menyimpan data mentah ke format CSV secara non-destruktif
    dengan menambahkan timestamp pada nama file.
    """
    if data.empty:
        logging.warning("Tidak ada data untuk disimpan.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp agar file lama tidak tertimpa
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"raw_stock_data_{current_time}.csv"
    file_path = os.path.join(output_dir, file_name)
    
    try:
        data.to_csv(file_path)
        logging.info(f"Data mentah tersimpan sukses di: {file_path}")
    except IOError as e:
        logging.error(f"Gagal menyimpan file CSV: {e}")

if __name__ == "__main__":
    # Parameter Eksekusi
    TARGET_TICKERS = ["BBCA.JK", "ADMR.JK", "^JKSE"]
    TIME_PERIOD = "3y"  # Menjamin data > 500 baris untuk LK-05
    RAW_DATA_FOLDER = "data/raw"
    
    # Menjalankan Pipeline Ingestion
    stock_dataframe = fetch_stock_data(TARGET_TICKERS, TIME_PERIOD)
    save_raw_data(stock_dataframe, RAW_DATA_FOLDER)