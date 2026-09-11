import yfinance as yf
print("Mencoba mengambil data saham BBCA.JK...")
data = yf.download("BBCA.JK", period="1d")
print(data)
print("\nSukses! Environment MLOps sudah siap.")
