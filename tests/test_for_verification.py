import csv
import time
import logging
import numpy as np
from typing import Dict, Any, Callable

######## SOLO PARA TEST UNITARIO
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
###########

from models.midbscan import MIDBSCAN
from data.arff_reader import ArffToMIData
from evaluation.evaluator import MILEvaluator
from preprocessing.scaler import MinMaxScaler, StandardScaler
from visualization.heatmap import plot_distance_heatmap
from optimization.knn_dist_eps import optimize_eps
from distances.hausdorff import hausdorff_distance
from distances.cauchy_schwarz import cauchy_schwarz_distance
from distances.distance_matrix import compute_distance_matrix
# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VerificationRunner")

# ─── Config ───────────────────────────────────────────────────────────────────
DATASETS_DIR    = "datasets"
HEATMAPS_DIR    = "results/output_heatmaps"
DEFAULT_MIN_PTS = 2
NUM_EPS_VALUES  = 7


# ─── Helpers ──────────────────────────────────────────────────────────────────
### REEMPLAZADO POR ALGORITMO DE OPTIMIZACIÓN KNN
def compute_eps_range(distance_matrix: np.ndarray, n: int = NUM_EPS_VALUES):
    """
    Calcula n valores de eps linealmente espaciados entre el percentil 5
    y el percentil 75 de las distancias inter-bags (triángulo superior).
    Los valores son siempre coherentes con el espacio ya normalizado.
    """
    upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    upper = upper[upper > 0]

    if len(upper) == 0:
        return list(np.linspace(0.05, 1.0, n).round(4))

    p_low  = float(np.percentile(upper, 5))
    p_high = float(np.percentile(upper, 75))

    if p_high <= p_low:
        p_high = float(np.max(upper))

    eps_values = np.linspace(p_low, p_high, n).round(4).tolist()
    logger.info(f"Rango de eps calculado: {eps_values}")
    return eps_values


# ─── Experimento ──────────────────────────────────────────────────────────────

def run_verification_experiment(
    filename: str,
    scaler_factory: Callable,
    min_pts: int = DEFAULT_MIN_PTS,
    metric: str = 'hausdorff',
) -> list[Dict[str, Any]]:
    """
    Workflow completo para un dataset:

      1. Carga el dataset.
      2. Normaliza con el scaler recibido por referencia (factory callable).
      3. Calcula la matriz de distancias, el rango de 7 eps y genera el heatmap
         (una sola vez: la matriz no varía entre iteraciones de eps).
      4. Ejecuta MIDBSCAN para cada valor de eps.
      5. Devuelve lista de filas de resultados (sin matriz de confusión).

    :param filename:        Nombre del archivo .arff dentro de DATASETS_DIR.
    :param scaler_factory:  Callable sin args que instancia un scaler nuevo
                            (ej: ``lambda: MinMaxScaler()``)
    :param min_pts:         Parámetro min_pts para MIDBSCAN.
    :param metric:          Métrica de distancia ('hausdorff' o 'cauchy_schwarz').
    :returns: Lista de dicts, uno por valor de eps probado.
    """

    file_path    = os.path.join(DATASETS_DIR, filename)
    dataset_name = os.path.splitext(filename)[0]
    rows: list[Dict[str, Any]] = []

    # ── PASO 1: Carga ──────────────────────────────────────────────────────
    logger.info(f"[{dataset_name}] PASO 1 — Cargando dataset...")
    loader    = ArffToMIData()
    full_data = loader.load(file_path, dataset_name=dataset_name)

    train_data, test_data = full_data.split_data(percentage_train=70, seed=42)
    logger.info(
        f"[{dataset_name}] Train: {train_data.get_num_bags()} bolsas | "
        f"Test: {test_data.get_num_bags()} bolsas"
    )

    # ── PASO 2: Normalización ──────────────────────────────────────────────
    scaler_instance = scaler_factory()
    scaler_name     = scaler_instance.__class__.__name__
    logger.info(f"[{dataset_name}] PASO 2 — Preprocesando con {scaler_name}...")

    train_scaled = scaler_instance.fit_transform(train_data)
    test_scaled  = scaler_instance.transform(test_data)

    # ── PASO 3: Seleccionar métrica de distancia ──────────────────────────
    metric_func_map = {
        'hausdorff': hausdorff_distance,
        'cauchy_schwarz': cauchy_schwarz_distance
    }
    distance_func = metric_func_map.get(metric, hausdorff_distance)
    logger.info(f"[{dataset_name}] Métrica de distancia seleccionada: {metric}")

    # ── PASO 4: Optimizar eps usando k-NN distance plot ─────────────────────
    logger.info(f"[{dataset_name}] PASO 4 — Optimizando eps mediante k-NN distance plot...")
    
    eps_optimal = optimize_eps(
        dataset=train_scaled,
        distance_func=distance_func,
        min_pts=min_pts,
        try_range=True,  # Explora k=1..20
        save_plots=True
    )
    
    logger.info(f"[{dataset_name}] Eps óptimo encontrado: {eps_optimal:.6f}")

    # ── PASO 5: Generar heatmap con la matriz de distancias ────────────────
    logger.info(f"[{dataset_name}] PASO 5 — Calculando matriz de distancias para heatmap...")
    
    dist_matrix = compute_distance_matrix(train_scaled.bags, distance_func, metric)
    
    bag_ids = [b.bag_id for b in train_scaled.bags]
    heatmap_filename = f"{dataset_name}_{scaler_name}_{metric}"
    saved_path = plot_distance_heatmap(
        distance_matrix=dist_matrix,
        bag_ids=bag_ids,
        title=f"{dataset_name}  |  {scaler_name}  |  {metric}",
        metric=metric,
        output_dir=f"{HEATMAPS_DIR}/{dataset_name}",
        filename=heatmap_filename,
        show=False,
    )
    logger.info(f"[{dataset_name}] Heatmap guardado → {saved_path}")

    # ── PASO 6: Ejecución con el eps óptimo ────────────────────────────────
    logger.info(f"[{dataset_name}] PASO 6 — Ejecutando MIDBSCAN con eps óptimo...")

# for run_idx, eps in enumerate(eps_values, start=1):
    start_time = time.time()

    row: Dict[str, Any] = {
        "Dataset":        dataset_name,
        "Scaler":         scaler_name,
        "Metric":         metric,
        "Epsilon":        eps_optimal,
        "MinPts":         min_pts,
        "Status":         "Failed",
        "Train_Bags":     train_data.get_num_bags(),
        "Test_Bags":      test_data.get_num_bags(),
        "Clusters":       0,
        "Noise_Pct":      0.0,
        "Precision":      0.0,
        "Recall":         0.0,
        "F1":             0.0,
        "Specificity":    0.0,
        "Execution_Time": 0.0,
        "Error_Msg":      "",
    }

    try:
        model = MIDBSCAN(epsilon=eps_optimal, min_pts=min_pts, metric=metric)
        model.fit(train_scaled)

        stats = model.get_statistics()
        row["Clusters"]  = stats.get("num_clusters", 0)
        row["Noise_Pct"] = round(stats.get("noise_percentage", 0), 2)

        predictions = model.predict(test_scaled)
        metrics = MILEvaluator.evaluate(
            test_scaled, predictions,
            title=f"Test {dataset_name} eps={eps_optimal}"
        )

        row["Precision"]   = round(metrics.get("Precision",   0), 4)
        row["Recall"]      = round(metrics.get("Recall",      0), 4)
        row["F1"]          = round(metrics.get("F1-Score",    0), 4)
        row["Specificity"] = round(metrics.get("Specificity", 0), 4)
        row["Status"]      = "Success"

    except Exception as e:
        logger.error(f"[{dataset_name}] Error en eps={eps_optimal}: {e}")
        row["Error_Msg"] = str(e)

    row["Execution_Time"] = round(time.time() - start_time, 2)
    rows.append(row)

    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(DATASETS_DIR):
        logger.error(f"El directorio '{DATASETS_DIR}' no existe.")
        return

    files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.arff')]
    if not files:
        logger.warning("No se encontraron archivos .arff en el directorio.")
        return

    logger.info(f"Encontrados {len(files)} dataset(s) para verificación.")

    # ── Configuración ─────────────────────────────────────────────────────
    # Cambia la lambda para usar otro scaler: lambda: StandardScaler()
    # scaler_factory = lambda: MinMaxScaler(feature_range=(0, 1))
    scaler_factory = lambda: StandardScaler()

    metric         = 'hausdorff'    # o 'cauchy_schwarz'
    min_pts        = DEFAULT_MIN_PTS

    all_results: list[Dict[str, Any]] = []

    for filename in files:
        logger.info(f"\n{'#'*60}\n  Procesando: {filename}\n{'#'*60}")
        rows = run_verification_experiment(
            filename=filename,
            scaler_factory=scaler_factory,
            min_pts=min_pts,
            metric=metric,
        )
        all_results.extend(rows)

    # ── PASO 5: Guardar CSV ────────────────────────────────────────────────
    scaler_name = scaler_factory().__class__.__name__
    output_csv  = f"results/verification_results_{scaler_name}_{metric}.csv"
    fieldnames  = [
        "Dataset", "Scaler", "Metric",
        "Epsilon", "MinPts", "Status", "Execution_Time",
        "Train_Bags", "Test_Bags",
        "Clusters", "Noise_Pct",
        "Precision", "Recall", "F1", "Specificity",
        "Error_Msg",
    ]

    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"Resultados guardados en '{output_csv}'")
    except IOError as e:
        logger.error(f"No se pudo escribir el archivo CSV: {e}")

    # ── Vista previa en consola ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(
        f"{'DATASET':<25} | {'EPS':>8} | {'STATUS':<10} | "
        f"{'F1':>8} | {'CLUSTERS':>8} | {'RUIDO %':>8}"
    )
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['Dataset']:<25} | {r['Epsilon']:>8} | {r['Status']:<10} | "
            f"{r['F1']:>8} | {r['Clusters']:>8} | {r['Noise_Pct']:>8}"
        )
    print("=" * 100)
    print(f"\nHeatmaps guardados en: {os.path.abspath(HEATMAPS_DIR)}")


if __name__ == "__main__":
    main()