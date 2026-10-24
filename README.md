# Proyecto Abalone - Predicción de Edad
Proyecto reproducible usando Python 3.15.1.
## Preprocesamiento
Para limpiar los datos, codificar la variable categórica y eliminar valores atípicos, ejecutar: 
`python src/preprocess.py`
## Entrenamiento del Modelo
Para entrenar el RandomForestRegressor, evaluar las métricas (MAE, RMSE, R2) y guardar el pipeline serializado, ejecutar:
`python src/train_model.py`
Los resultados se registran en la carpeta `results/`.
## Resultados
* Experimento v2 (RF n_estimators=200, max_depth=10): Completado.