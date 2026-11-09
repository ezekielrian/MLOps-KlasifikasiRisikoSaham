import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

RANDOM_SEED = 42

def prepare_data(filepath: str):
    """
    Membaca data historis saham dan menyiapkan fitur (X) serta label target (y).
    
    Args:
        filepath (str): Path direktori menuju file CSV data bersih.
        
    Returns:
        tuple: Kumpulan data latih dan uji (X_train, X_test, y_train, y_test).
    """
    df = pd.read_csv(filepath)
    
    df = df.iloc[2:].reset_index(drop=True)
    
    cols_to_convert = ['Open', 'High', 'Low', 'Volume', 'Close']
    df[cols_to_convert] = df[cols_to_convert].astype(float)
    
    # Rekayasa fitur target klasifikasi (1: Naik/Aman, 0: Turun/Bahaya)
    if 'Target' not in df.columns:
        df['Target'] = (df['Close'] > df['Open']).astype(int)
    
    features = ['Open', 'High', 'Low', 'Volume']
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test, n_estimators: int, max_depth: int):
    """
    Melatih model RandomForestClassifier dan melacak eksperimen menggunakan MLflow.
    
    Args:
        X_train (DataFrame): Data fitur untuk pelatihan.
        X_test (DataFrame): Data fitur untuk pengujian.
        y_train (Series): Data label untuk pelatihan.
        y_test (Series): Data label untuk pengujian.
        n_estimators (int): Jumlah tree dalam Random Forest.
        max_depth (int): Kedalaman maksimum setiap tree.
    """
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_SEED
        )
        
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        # Pencatatan Parameter Eksperimen
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Pencatatan Metrik Evaluasi
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Pencatatan Artefak Model (Perbaikan dari Codex)
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="random_forest_model"
        )
        
        print(f"Run tersimpan -> n_estimators: {n_estimators:3}, max_depth: {max_depth:2} | Akurasi: {accuracy:.4f} | F1-Score: {f1:.4f}")

if __name__ == "__main__":
    mlflow.set_experiment("Klasifikasi_Saham_V3")
    data_path = "data/processed/processed_stock_data.csv"
    
    try:
        X_train, X_test, y_train, y_test = prepare_data(data_path)
        
        print("Memulai Eksekusi MLflow...\n")
        
        # Skenario Eksperimen (Hyperparameter Tuning)
        train_model(X_train, X_test, y_train, y_test, n_estimators=50, max_depth=5)
        train_model(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=10)
        train_model(X_train, X_test, y_train, y_test, n_estimators=200, max_depth=15)
        
        print("\nSeluruh eksperimen telah selesai dan tercatat di MLflow.")
        
    except FileNotFoundError:
        print(f"Error: File {data_path} tidak ditemukan. Pastikan path benar atau jalankan 'dvc pull'.")