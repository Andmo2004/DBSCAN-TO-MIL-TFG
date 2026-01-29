import os
import sys
import csv
import time
import logging
from typing import Dict, Any

# Obtiene la ruta absoluta de ESTE script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Obtiene la ruta del directorio PADRE (ej: .../proyecto/)
project_root = os.path.dirname(current_dir)
# Añade el padre al path de Python para que encuentre 'data', 'models', etc.
sys.path.append(project_root)

# Importaciones de tu framework
from data.arff_reader import ArffToMIData
from models.midbscan import MIDBSCAN
from evaluation.evaluator import MILEvaluator
# [NUEVO] Importamos el preprocesamiento
from data.preprocessing import MinMaxScaler, StandardScaler

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ExperimentRunner")

# Configuración Global
DATASETS_DIR = "datasets"
OUTPUT_CSV = "results_summary_scaled" # Cambié el nombre para no sobrescribir el anterior

# [IMPORTANTE] Ajuste de Parámetros para datos escalados
# Al escalar a [0, 1], las distancias serán muy pequeñas.
# Un epsilon de 1000 agruparía todo en un solo cluster.
# Probamos con 0.5 (la mitad del rango normalized) como punto de partida.
DEFAULT_EPSILON = 0.5  
DEFAULT_MIN_PTS = 2

def run_single_experiment(filename: str, scaler_type: str) -> Dict[str, Any]:
    """
    Carga un dataset, ESCALA, entrena, predice y evalúa.
    """
    file_path = os.path.join(DATASETS_DIR, filename)
    dataset_name = os.path.splitext(filename)[0]
    
    start_time = time.time()
    result_row = {
        "Dataset": dataset_name,
        "Status": "Failed",
        "Train_Bags": 0,
        "Test_Bags": 0,
        "Clusters": 0,
        "Noise_Pct": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
        "Specificity": 0.0,
        "Error_Msg": ""
    }

    try:
        # 1. Carga de Datos
        logger.info(f"[{dataset_name}] Cargando datos...")
        loader = ArffToMIData()
        full_data = loader.load(file_path, dataset_name=dataset_name)
        
        # 2. División Train/Test (70% Train, 30% Test)
        train_data, test_data = full_data.split_data(percentage_train=70, seed=42)
        
        result_row["Train_Bags"] = train_data.get_num_bags()
        result_row["Test_Bags"] = test_data.get_num_bags()

        # -------------------------------------------------------------
        # [NUEVO] 2.5. Escalado / Normalización
        # -------------------------------------------------------------
        logger.info(f"[{dataset_name}] Escalando datos (MinMax 0-1)...")
        
        # Elegimos MinMaxScaler para acotar distancias entre 0 y 1.
        # (Si prefieres Z-Score, cambia a: scaler = StandardScaler())
        if scaler_type == "MinMax":
            scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == "Standard":
            scaler = StandardScaler()
        else:
            raise TypeError("No se ha seleccionado ningun tipo de escalado")
        
        # A) FIT (Aprender) SOLO con los datos de entrenamiento
        # B) TRANSFORM (Aplicar) a entrenamiento
        train_scaled = scaler.fit_transform(train_data)
        
        # C) TRANSFORM (Aplicar) a test usando los parámetros aprendidos de train
        test_scaled = scaler.transform(test_data)
        # -------------------------------------------------------------

        # 3. Configuración y Entrenamiento del Modelo
        # Usamos los datos escalados (train_scaled)
        logger.info(f"[{dataset_name}] Entrenando MIDBSCAN (Eps={DEFAULT_EPSILON})...")
        model = MIDBSCAN(epsilon=DEFAULT_EPSILON, min_pts=DEFAULT_MIN_PTS)
        model.fit(train_scaled)
        
        # Estadísticas del modelo
        stats = model.get_statistics()
        result_row["Clusters"] = stats.get("num_clusters", 0)
        result_row["Noise_Pct"] = round(stats.get("noise_percentage", 0), 2)

        # 4. Predicción
        # Usamos los datos de test escalados (test_scaled)
        predictions = model.predict(test_scaled)
        
        # 5. Evaluación
        logger.info(f"[{dataset_name}] Evaluando resultados...")
        # Evaluamos sobre test_scaled (las etiquetas reales se conservan en el objeto)
        metrics = MILEvaluator.evaluate(test_scaled, predictions, title=f"Test {dataset_name}")
        
        # Actualizar fila de resultados
        result_row["Precision"] = round(metrics.get("Precision", 0), 4)
        result_row["Recall"] = round(metrics.get("Recall", 0), 4)
        result_row["F1"] = round(metrics.get("F1-Score", 0), 4)
        result_row["Specificity"] = round(metrics.get("Specificity", 0), 4)
        result_row["Status"] = "Success"

    except Exception as e:
        logger.error(f"[{dataset_name}] Error: {e}")
        # import traceback
        # traceback.print_exc() # Descomentar para ver el error completo en consola
        result_row["Error_Msg"] = str(e)
    
    result_row["Execution_Time"] = round(time.time() - start_time, 2)
    return result_row

def main():

    # Verificar directorio
    if not os.path.exists(DATASETS_DIR):
        logger.error(f"El directorio '{DATASETS_DIR}' no existe.")
        return

    # Obtener archivos .arff
    files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.arff')]
    
    if not files:
        logger.warning("No se encontraron archivos .arff en el directorio.")
        return

    logger.info(f"Encontrados {len(files)} datasets para procesar.")
    results = []

    # Ejecutar experimentos
    scaler_type = "MinMax"
    for i, filename in enumerate(files):
        logger.info(f"--- Procesando {i+1}/{len(files)}: {filename} ---")
        row = run_single_experiment(filename, scaler_type)
        results.append(row)

    # Guardar a CSV
    fieldnames = [
        "Dataset", "Status", "Execution_Time", 
        "Train_Bags", "Test_Bags", 
        "Clusters", "Noise_Pct", 
        "Precision", "Recall", "F1", "Specificity", 
        "Error_Msg"
    ]
    
    try:
        with open(f'{OUTPUT_CSV}_{scaler_type}.csv', mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"\nResumen completado. Resultados guardados en '{OUTPUT_CSV}'")
        
        # Imprimir vista previa en consola
        print("\n" + "="*90)
        print(f"{'DATASET':<25} | {'STATUS':<10} | {'F1-SCORE':<10} | {'CLUSTERS':<8} | {'RUIDO %':<8}")
        print("-" * 90)
        for r in results:
            print(f"{r['Dataset']:<25} | {r['Status']:<10} | {r['F1']:<10} | {r['Clusters']:<8} | {r['Noise_Pct']}")
        print("="*90)

    except IOError as e:
        logger.error(f"No se pudo escribir el archivo CSV: {e}")

if __name__ == "__main__":
    main()