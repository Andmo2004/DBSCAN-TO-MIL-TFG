import os
import sys
import pandas as pd
from tabulate import tabulate  

# Obtiene la ruta absoluta de ESTE script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Obtiene la ruta del directorio PADRE (ej: .../proyecto/)
project_root = os.path.dirname(current_dir)
# Añade el padre al path de Python para que encuentre 'data', 'models', etc.
sys.path.append(project_root)

from data.arff_reader import ArffToMIData
from models.midbscan import MIDBSCAN
from evaluation.evaluator import MILEvaluator
from data.preprocessing import MinMaxScaler 

# Configuración de Logging
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("DummyTester")

# ---------------------------------------------------------
#   SCRIPT DE EVALUACIÓN
# ---------------------------------------------------------
def run_grid_search():
    # Asegúrate de que este archivo exista o usa el generador previo
    filename = "datasets/simple_dummy.arff" 
    
    # Define el nombre del archivo de salida
    output_csv = "dummy_grid_results.csv"

    # 1. Cargar Datos
    try:
        loader = ArffToMIData()
        dataset = loader.load(filename)
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo '{filename}'. Asegúrate de generarlo primero.")
        return

    # 2. Escalar Datos
    scaler = MinMaxScaler(feature_range=(0, 10)) 
    dataset_scaled = scaler.fit_transform(dataset)
    
    print("\nDatos escalados al rango [0, 10].")
    print(f"Total Bolsas: {dataset.get_num_bags()}")
    
    # 3. Definir Grid de Parámetros
    eps_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0] 
    min_pts_values = [2, 3, 4]
    
    results = []

    print("\nIniciando Grid Search...")
    print("-" * 80)
    
    for min_pts in min_pts_values:
        for eps in eps_values:
            # A) Entrenar
            model = MIDBSCAN(epsilon=eps, min_pts=min_pts)
            model.fit(dataset_scaled)
            
            # B) Evaluar
            predictions = model.predict(dataset_scaled)
            
            # Obtener estadísticas internas
            stats = model.get_statistics()
            n_clusters = stats['num_clusters']
            noise_pct = stats['noise_percentage']
            
            # Calcular métricas supervisadas
            metrics = MILEvaluator.evaluate(dataset_scaled, predictions, title="")            
            
            # Interpretación simple
            interp = "Malo"
            if n_clusters == 2 and metrics['F1-Score'] > 0.8:
                interp = "EXCELENTE"
            elif n_clusters > 2:
                interp = "Sobre segmentado." 
            elif n_clusters == 1:
                interp = "Sub segmentado." 
            elif n_clusters == 0:
                interp = "Todo Ruido"

            # Guardamos los datos puros en el diccionario
            results.append({
                "Eps": eps,
                "MinPts": min_pts,
                "Clusters": n_clusters,
                "Ruido_Pct": round(noise_pct, 2), # Nombre sin espacios para CSV fácil
                "Precision": round(metrics.get('Precision', 0), 4),
                "Recall": round(metrics.get('Recall', 0), 4),
                "Specificity": round(metrics.get('Specificity', 0), 4),
                "F1_Score": round(metrics['F1-Score'], 4),
                "Interpretacion": interp
            })

    # 4. Imprimir en Consola (Formato Tabla)
    print("\n--- RESULTADOS ---")
    try:
        print(tabulate(results, headers="keys", tablefmt="github"))
    except ImportError:
        print(results)

    print("-" * 80)
    
    # 5. GUARDAR A CSV (NUEVO BLOQUE)
    # ---------------------------------------------------------
    try:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n[OK] Resultados exportados exitosamente a: {os.path.abspath(output_csv)}")
    except Exception as e:
        print(f"\n[ERROR] No se pudo guardar el CSV: {e}")
    # ---------------------------------------------------------

    print("\nGUÍA DE INTERPRETACIÓN:")
    print("1. EXCELENTE: Encuentra 2 clusters (Positivo y Negativo) y el F1 es alto.")
    print("2. Todo Ruido: Epsilon es muy pequeño.")
    print("3. Subsed (1 Cluster): Epsilon muy grande.")

if __name__ == "__main__":
    run_grid_search()