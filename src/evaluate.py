import mlflow
from mlflow.tracking import MlflowClient

def evaluate_and_promote(threshold=0.50):
    client = MlflowClient()
    experiment_name = "Klasifikasi_Saham_V3"
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Eksperimen {experiment_name} tidak ditemukan.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        print("Tidak ada run yang ditemukan.")
        return

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    accuracy = latest_run.data.metrics.get("accuracy", 0)

    print(f"Run ID Terakhir: {run_id} | Akurasi: {accuracy:.4f}")

    if accuracy >= threshold:
        print(f"LULUS! Akurasi {accuracy:.4f} >= Threshold {threshold}.")
        model_name = "Model_Risiko_Saham_Ezekiel"
        model_uri = f"runs:/{run_id}/random_forest_model"
        
        registered_model = mlflow.register_model(model_uri, model_name)
        
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="Staging"
        )
        print(f"SUKSES: Model versi {registered_model.version} dipromosikan ke Staging!")
    else:
        print(f"GAGAL: Akurasi {accuracy:.4f} di bawah threshold {threshold}. Model ditolak.")

if __name__ == "__main__":
    evaluate_and_promote()