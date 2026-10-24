import pandas as pd
import json
from datetime import datetime

def main():
    # Rutas
    input_path = "data/raw/abalone_raw.csv"
    output_path = "data/processed/abalone_clean.csv"
    log_path = "data/data_log.json"

    # 1. Cargar datos (asignando los nombres de las columnas manualmente)
    column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 
                    'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
    df = pd.read_csv(input_path, names=column_names)

    # Nueva columna Age (age = rings + 1.5)
    df['Age'] = df['Rings'] + 1.5

    # Codificacion One-Hot 'Sex'
    df = pd.get_dummies(df, columns=['Sex'], prefix='sex', dtype=int)

    # Revisión de valores atípicos 
    # Criterio: En Abalone existen registros erróneos con altura 0, y valores extremos > 0.4
    initial_count = len(df)
    df = df[(df['Height'] > 0) & (df['Height'] <= 0.4)]
    outliers_removed = initial_count - len(df)

    # Guardar dataset limpio
    df.to_csv(output_path, index=False)

    # Registrar operaciones en el log
    log_data = {
        "version": "v1",
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "source": input_path,
        "transformations": [
            "added derived feature 'age = rings + 1.5'",
            "encoded sex with one-hot encoding (sex_F, sex_I, sex_M)",
            f"removed {outliers_removed} outliers with height <= 0 or height > 0.4"
        ]
    }

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)

    print(f"Preprocesamiento finalizado. Se eliminaron {outliers_removed} valores atípicos.")

if __name__ == "__main__":
    main()