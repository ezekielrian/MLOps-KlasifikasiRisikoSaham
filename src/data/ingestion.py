import yfinance as yf
import os

tickers = ["BBCA.JK", "ADMR.JK", "^JKSE"]
output_dir = "data/raw"

os.makedirs(output_dir, exist_ok=True)

print("=== Memulai Ingestion Data (PoC LK-03) ===\n")

for ticker in tickers:
    print(f"Menarik data {ticker}...")
    data = yf.download(ticker, period="5d")
    
    if not data.empty:
        file_path = f"{output_dir}/{ticker}_raw.csv"
        data.to_csv(file_path)
        print(f"[SUCCESS] Data disimpan ke: {file_path}\n")
    else:
        print(f"[FAILED] Data {ticker} tidak ditemukan.\n")

print("=== Proses Ingestion Selesai ===")