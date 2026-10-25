import pandas as pd
import json
import csv
import os
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # Configuración de rutas y semilla
    SEED = 42
    DATA_PATH = "data/processed/abalone_clean.csv"
    MODEL_PATH = "models/model_v3.pkl"
    METRICS_PATH = "results/metrics_v3.json"
    LOG_PATH = "results/experiments_log.csv"

    # 1. Cargar datos limpios
    df = pd.read_csv(DATA_PATH)

    # 2. Separar características (X) y variable objetivo (y)
    # Eliminamos 'age' (objetivo) y 'Rings' (para evitar data leakage)
    X = df.drop(columns=['Age', 'Rings'])
    y = df['Age']

    # 3. Dividir en entrenamiento y test fijando la semilla
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # 4. Pipeline: Escalado + Modelo (ajusta el scaler SOLO con X_train)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED))
    ])
    pipeline.fit(X_train, y_train)

    # 5. Evaluación y métricas
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # 6. Guardar modelo con compresión 3
    joblib.dump(pipeline, MODEL_PATH, compress=3)

    # 7. Guardar métricas en JSON
    hiperparametros = {"n_estimators": 200, "max_depth": 10, "random_state": SEED, "model": "RandomForest"}
    
    with open(METRICS_PATH, 'w') as f:
        json.dump({
            "hyperparameters": hiperparametros,
            "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
            "seed": SEED
        }, f, indent=4)

    # 8. Registrar experimento en CSV
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id_experimento", "fecha_hora", "version_modelo", "hiperparametros", "MAE", "RMSE", "R2"])
        
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([exp_id, fecha_hora, "v3", json.dumps(hiperparametros), round(mae, 4), round(rmse, 4), round(r2, 4)])

    print(f"Modelo entrenado y guardado. R2: {r2:.4f}, RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()