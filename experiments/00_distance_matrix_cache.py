"""
00_distance_matrix_cache.py

Script para precomputar las matrices de distancia y almacenarlas en la caché persistente (disco).
Esto permite ahorrar muchísimo tiempo en las fases siguientes del experimento.
"""

import os
import sys
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config.settings import DATASETS_CONFIG
from data.midata import MIData
from preprocessing.scaler import MinMaxScaler, StandardScaler
from distances.matrix_cache import global_persistent_cache
from distances.hausdorff import hausdorff_distance, hausdorff_distance_min, hausdorff_distance_avg
from distances.probability_distribution import cauchy_schwarz_distance, earth_movers_distance, mahalanobis_distance

logging.basicConfig(level=logging.INFO, format="%(message)s")

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

def precompute_matrices():
    print("=" * 80)
    print("PRECOMPUTANDO MATRICES DE DISTANCIA (CACHÉ PERSISTENTE)")
    print("=" * 80)
    
    for config in DATASETS_CONFIG:
        dataset_name = config["dataset_name"]
        arff_name = config["arff_name"]
        
        print(f"\n► Procesando Dataset: {dataset_name}")
        path = os.path.join("datasets", f"{arff_name}.arff")
        if not os.path.exists(path):
            print(f"  [!] No se encontró el archivo: {path}. Omitiendo.")
            continue
            
        dataset_full = MIData.from_arff(path)
        train_data, _ = dataset_full.split_data(percentage_train=70, seed=42)
        
        available_metrics = list(DISTANCES.keys())
        if dataset_name in ["Harddrive1", "Thioredoxin", "Newsgroups1"]:
            if "earth_movers" in available_metrics:
                available_metrics.remove("earth_movers")
        
        for scaler_name, ScalerClass in SCALERS.items():
            scaler = ScalerClass()
            scaled_train = scaler.fit_transform(train_data)
            
            for metric_name in available_metrics:
                metric_func = DISTANCES[metric_name]
                
                # Precompute for train split
                global_persistent_cache.get(
                    dataset_name=dataset_name,
                    split="train",
                    scaler_name=scaler_name,
                    metric_name=metric_name,
                    bags=scaled_train.bags,
                    metric_func=metric_func
                )
                
        # Liberar memoria de la caché local para el siguiente dataset (ya está guardado en disco)
        global_persistent_cache.clear_memory()

    print("\n" + "=" * 80)
    print("PRECOMPUTACIÓN COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    precompute_matrices()
