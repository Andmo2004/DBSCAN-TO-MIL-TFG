"""
optimization/best_params.py

Búsqueda de hiperparámetros óptimos para MIDBSCAN usando Optuna.
Optimiza: Scaler, Métrica de Distancia, min_pts y eps.

Para que la optimización de eps sea agnóstica a la escala de la distancia,
Optuna busca un 'percentil' (eps_percentile). El valor real absoluto de eps 
se calcula usando la matriz de distancias precomputada.
"""

import os
import sys
import csv
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Tuple

# Evitamos que Optuna llene la consola con demasiada info por defecto
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from data.midata import MIData
from models.midbscan import MIDBSCAN
from preprocessing.scaler import MinMaxScaler, StandardScaler
from distances.distance_matrix import compute_distance_matrix

# Funciones de evaluación
from evaluation.scoring import detect_imbalance_ratio, score_labels

# Métricas de distancia
from distances.hausdorff import hausdorff_distance, hausdorff_distance_min, hausdorff_distance_avg
from distances.probability_distribution import cauchy_schwarz_distance, earth_movers_distance, mahalanobis_distance

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCALERS = {
    "MinMaxScaler": MinMaxScaler,
    "StandardScaler": StandardScaler
}

DISTANCES = {
    "hausdorff": hausdorff_distance,
    "hausdorff_min": hausdorff_distance_min,
    "hausdorff_avg": hausdorff_distance_avg,
    "cauchy_schwarz": cauchy_schwarz_distance,
    "earth_movers": earth_movers_distance,
    "mahalanobis": mahalanobis_distance
}

# --- NUEVA LÓGICA DE WARM STARTING ---
# Diccionario para almacenar configuraciones previas óptimas (Best Known Configurations).
KNOWN_BESTS = {
    "musk1":              {"scaler": "MinMaxScaler",   "metric": "hausdorff",      "min_pts": 2,  "eps_abs": 2.1673},
    "musk2":              {"scaler": "MinMaxScaler",   "metric": "cauchy_schwarz", "min_pts": 3,  "eps_abs": 0.02026},
    "ImageElephant":      {"scaler": "MinMaxScaler",   "metric": "cauchy_schwarz", "min_pts": 2,  "eps_abs": 0.11840},
    "BirdsChestnut":      {"scaler": "StandardScaler", "metric": "cauchy_schwarz", "min_pts": 10, "eps_abs": 0.2988},
    "BirdsHammonds":      {"scaler": "MinMaxScaler",   "metric": "cauchy_schwarz", "min_pts": 2,  "eps_abs": 0.00565},
    "Harddrive1":         {"scaler": "MinMaxScaler",   "metric": "cauchy_schwarz", "min_pts": 3,  "eps_abs": 0.003467},
    "mutagenesis_atoms":  {"scaler": "StandardScaler", "metric": "hausdorff",      "min_pts": 3,  "eps_abs": 0.4748},
    "mutagenesis_chains": {"scaler": "MinMaxScaler",   "metric": "cauchy_schwarz", "min_pts": 3,  "eps_abs": 0.006638},
    "Newsgroups1":        {"scaler": "StandardScaler", "metric": "hausdorff",      "min_pts": 2,  "eps_abs": 50.434},
    "Thioredoxin":        {"scaler": "MinMaxScaler",   "metric": "cauchy_schwarz", "min_pts": 2,  "eps_abs": 0.001185},
}

DATASETS_CONFIG = [
    {"dataset_name": "musk1",              "arff_name": "musk1"},
    {"dataset_name": "musk2",              "arff_name": "musk2"},
    {"dataset_name": "ImageElephant",      "arff_name": "ImageElephant"},
    {"dataset_name": "BirdsChestnut",      "arff_name": "BirdsChestnut-backedChickadee"},
    {"dataset_name": "BirdsHammonds",      "arff_name": "BirdsHammondsFlycatcher"},
    {"dataset_name": "Harddrive1",         "arff_name": "Harddrive1"},
    {"dataset_name": "mutagenesis_atoms",  "arff_name": "mutagenesis3_atoms"},
    {"dataset_name": "mutagenesis_chains", "arff_name": "mutagenesis3_chains"},
    {"dataset_name": "Newsgroups1",        "arff_name": "Newsgroups1"},
    {"dataset_name": "Thioredoxin",        "arff_name": "Thioredoxin"},
]

class DistanceMatrixCache:
    """
    Caché para almacenar las matrices de distancias y evitar recalcularlas 
    en cada trial de Optuna para la misma combinación de dataset + scaler + métrica.
    """
    def __init__(self):
        self._cache = {}

    def get(self, dataset_name: str, split: str, scaler_name: str, metric_name: str, bags: list) -> np.ndarray:
        key = (dataset_name, split, scaler_name, metric_name)
        if key not in self._cache:
            metric_func = DISTANCES[metric_name]
            logger.warning(f"[{dataset_name}] Calculando matriz de distancias para {scaler_name} + {metric_name}...")
            self._cache[key] = compute_distance_matrix(bags, metric_func, metric_name)
        return self._cache[key]

# Instancia global de caché
global_cache = DistanceMatrixCache()


def create_objective(dataset: MIData, dataset_name: str):
    """
    Fábrica de la función objetivo para Optuna asociada a un dataset.
    """
    imbalance_ratio = detect_imbalance_ratio(dataset)
    scaled_datasets_cache = {}

    # 1. Definir qué métricas están permitidas según el dataset
    available_metrics = list(DISTANCES.keys())
    
    # 2. Excluimos EMD de los datasets computacionalmente intratables o muy densos
    # (Basado en el análisis de la media y el máximo de instancias por bolsa)
    if dataset_name in ["Harddrive1", "Thioredoxin", "Newsgroups1"]:
        if "earth_movers" in available_metrics:
            available_metrics.remove("earth_movers")
            logger.warning(f"  [!] Excluyendo 'earth_movers' para {dataset_name} por coste computacional.")
    
    def objective(trial: optuna.Trial) -> float:
        scaler_name = trial.suggest_categorical("scaler", list(SCALERS.keys()))
        metric_name = trial.suggest_categorical("metric", available_metrics)
        min_pts = trial.suggest_int("min_pts", 2, 20)
        
        # El usuario sugirió buscar eps_percentile entre 1.0 y 40.0
        eps_percentile = trial.suggest_float("eps_percentile", 1.0, 40.0)

        # 1. Escalar el dataset (con caché)
        if scaler_name not in scaled_datasets_cache:
            scaler = SCALERS[scaler_name]()
            scaled_datasets_cache[scaler_name] = scaler.fit_transform(dataset)
        scaled_dataset = scaled_datasets_cache[scaler_name]

        # 2. Obtener matriz de distancias desde la caché
        dist_matrix = global_cache.get(
            dataset_name=dataset_name, 
            split="train", 
            scaler_name=scaler_name, 
            metric_name=metric_name, 
            bags=scaled_dataset.bags
        )

        # 3. Calcular eps absoluto a partir del percentil
        # Se toma la diagonal superior de la matriz de distancias (excluyendo la diagonal principal)
        # y se busca el percentil de los valores positivos.
        upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        upper_positive = upper[upper > 0]
        
        if len(upper_positive) == 0:
             raise optuna.exceptions.TrialPruned("Todos los valores de distancia son 0.")
             
        eps_absolute = float(np.percentile(upper_positive, eps_percentile))
        
        # 4. Entrenar modelo
        try:
            model = MIDBSCAN(epsilon=eps_absolute, min_pts=min_pts, metric=metric_name)
            # Inyección directa de la matriz para no recalcular
            model._distance_matrix = dist_matrix 
            model.fit(scaled_dataset)
            
            # Extraer estadísticas para el registro
            stats = model.get_statistics()
            noise_pct = stats.get("noise_percentage", 0) / 100.0

            # 5. Evaluar (score_labels ya aplica su propia penalización por ruido internamente)
            score = score_labels(scaled_dataset, model.labels, imbalance_ratio=imbalance_ratio)

            # Guardamos variables de interés para que queden registradas en el study de Optuna
            trial.set_user_attr("eps_absolute", eps_absolute)
            trial.set_user_attr("clusters", model.cluster_count)
            trial.set_user_attr("noise_pct", round(noise_pct * 100, 1))

            # Penalizamos la puntuación a cero si todo el dataset resulta ser ruido
            if model.cluster_count == 0:
                return 0.0
                
            return score
            
        except Exception as e:
            logger.warning(f"Trial falló: {e}")
            raise optuna.exceptions.TrialPruned()

    return objective


def run_optuna_search(n_trials: int = 100):
    os.makedirs("results", exist_ok=True)
    
    results = []
    studies = {}
    
    print(f"\n{'='*70}")
    print(f"  INICIANDO BÚSQUEDA OPTUNA ({n_trials} trials por dataset)")
    print(f"{'='*70}")

    for config in DATASETS_CONFIG:
        dataset_name = config["dataset_name"]
        arff_name = config["arff_name"]
        
        print(f"\n► Procesando Dataset: {dataset_name}...")
        global_cache._cache.clear()
        path = os.path.join("datasets", f"{arff_name}.arff")
        if not os.path.exists(path):
            print(f"  [!] No se encontró el archivo: {path}. Omitiendo.")
            continue
            
        # Cargar y dividir de manera idéntica al main/test_full_eval
        dataset_full = MIData.from_arff(path)
        train_data, _ = dataset_full.split_data(percentage_train=70, seed=42)
        
        study_name = f"midbscan_optuna_{dataset_name}"

        # 1. Creamos un "sampler" (muestreador) de Optuna fijando la semilla
        sampler = optuna.samplers.TPESampler(seed=42)
        
        # 2. Le pasamos el sampler al estudio
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler  
        )
        
        if dataset_name in KNOWN_BESTS:
            kb = KNOWN_BESTS[dataset_name]
            
            # 1. Escalar los datos temporalmente para sacar la matriz
            scaler_class = SCALERS[kb["scaler"]]
            scaled_dataset = scaler_class().fit_transform(train_data)
            
            # 2. Obtener la matriz de distancias desde tu caché
            dist_matrix = global_cache.get(
                dataset_name=dataset_name, 
                split="train", 
                scaler_name=kb["scaler"], 
                metric_name=kb["metric"], 
                bags=scaled_dataset.bags
            )
            
            # 3. Calcular a qué percentil equivale exactamente tu eps_abs
            upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            upper_positive = upper[upper > 0]
            
            if len(upper_positive) > 0:
                # Calculamos el porcentaje de distancias que son menores o iguales a tu eps_abs
                exact_percentile = float(np.mean(upper_positive <= kb["eps_abs"]) * 100.0)
                
                # 4. Inyectar en Optuna
                study.enqueue_trial({
                    "scaler": kb["scaler"],
                    "metric": kb["metric"],
                    "min_pts": kb["min_pts"],
                    "eps_percentile": exact_percentile
                })
                print(f"  [+] Inyectando solución previa (Warm Start): eps_percentile={exact_percentile:.2f}%")

        # Envoltorio de la función objetivo
        objective = create_objective(train_data, dataset_name)
        
        # Optimizar
        try:
            study.optimize(
                objective, 
                n_trials=n_trials, 
                n_jobs=1, # Para prevenir recálculo paralelo de las mismas matrices
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print(f"  [!] Interrumpido por el usuario en el dataset {dataset_name}.")
            break
            
        if not study.best_trials:
            print("  [!] Ningún trial exitoso para este dataset.")
            continue
            
        best = study.best_trial
        eps_absolute = best.user_attrs.get("eps_absolute", 0.0)
        clusters = best.user_attrs.get("clusters", 0)
        noise_pct = best.user_attrs.get("noise_pct", 0.0)
        
        print(f"  >> Mejor F1 Score : {best.value:.4f}")
        print(f"  >> Scaler         : {best.params['scaler']}")
        print(f"  >> Distancia      : {best.params['metric']}")
        print(f"  >> eps_percentile : {best.params['eps_percentile']:.2f}% -> eps_abs: {eps_absolute:.6f}")
        print(f"  >> min_pts        : {best.params['min_pts']}")
        print(f"  >> Clusters Hall. : {clusters}")
        print(f"  >> Ruido (%)      : {noise_pct}%")
        
        results.append({
            "dataset": dataset_name,
            "best_score": round(best.value, 4),
            "scaler": best.params["scaler"],
            "metric": best.params["metric"],
            "min_pts": best.params["min_pts"],
            "eps_percentile": round(best.params["eps_percentile"], 2),
            "eps_absolute": round(eps_absolute, 6),
            "clusters": clusters,
            "noise_pct": noise_pct
        })
        
        studies[dataset_name] = study
        
    # Guardar todos los mejores en CSV
    if results:
        ts = datetime.now().strftime("%d%m%Y%H%M")
        csv_file = os.path.join("results", f"optuna_best_params_{ts}.csv")
        
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "dataset", "best_score", "scaler", "metric", "min_pts", 
                "eps_percentile", "eps_absolute", "clusters", "noise_pct"
            ])
            writer.writeheader()
            writer.writerows(results)
            
        print(f"\n{'='*70}")
        print(f"  BÚSQUEDA COMPLETADA.")
        print(f"  Mejores configuraciones guardadas en: {csv_file}")
        print(f"{'='*70}\n")
    else:
        print("\n  [!] No se generaron resultados para guardar.\n")

    return results, studies


if __name__ == '__main__':
    run_optuna_search(n_trials=100)
