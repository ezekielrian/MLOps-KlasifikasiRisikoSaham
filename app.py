import pandas as pd
import mlflow.pyfunc
from flask import Flask, request, jsonify

app = Flask(__name__)

# Konfigurasi Model
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "Model_Risiko_Saham_Ezekiel"
STAGE = "Production"
MODEL_URI = f"models:/{MODEL_NAME}/{STAGE}"

print(f"[*] Menarik model MLflow dari: {MODEL_URI} ...")
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("[*] Model berhasil dimuat dan siap melayani inferensi!")
except Exception as e:
    print(f"[!] Gagal memuat model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk melakukan inferensi prediksi risiko saham.
    Menerima input JSON dengan key 'features' yang berisi list of dictionaries.
    Memastikan urutan matriks fitur sesuai dengan fase pelatihan (LK-06).
    """
    # Proteksi jika model gagal dimuat saat server start
    if not model:
        return jsonify({"status": "error", "message": "Model tidak tersedia di server"}), 500

    try:
        # 1. Parsing payload JSON
        request_data = request.get_json()
        if not request_data or 'features' not in request_data:
            return jsonify({"status": "error", "message": "Payload tidak valid. Harus memiliki key 'features'"}), 400
        
        raw_data = request_data['features']
        
        # 2. Konversi ke DataFrame
        df = pd.DataFrame(raw_data)
        
        # 3. Validasi ketat nama dan urutan kolom (First-principles check)
        expected_columns = ['Open', 'High', 'Low', 'Volume']
        
        for col in expected_columns:
            if col not in df.columns:
                return jsonify({"status": "error", "message": f"Fitur '{col}' wajib ada di dalam request!"}), 400
                
        # Memastikan urutan kolom mengunci presisi model
        df_prepared = df[expected_columns]
        
        # 4. Eksekusi prediksi
        predictions = model.predict(df_prepared)
        
        # 5. Format dan kembalikan response
        return jsonify({
            "status": "success",
            "model_version": STAGE,
            "prediction": predictions.tolist()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Mengikat pada 0.0.0.0 agar port dapat diakses saat dibungkus Docker
    app.run(host="0.0.0.0", port=5000, debug=False)