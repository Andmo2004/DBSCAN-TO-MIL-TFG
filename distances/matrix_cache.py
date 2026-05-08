import os
import numpy as np
import logging
from config.settings import BASE_DIR
from distances.distance_matrix import compute_distance_matrix

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(BASE_DIR, "results", "distance_matrices")

class PersistentDistanceMatrixCache:
    """
    Caché para almacenar matrices de distancias en disco y evitar recalcularlas.
    """
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._memory_cache = {}

    def get(self, dataset_name: str, split: str, scaler_name: str, metric_name: str, bags: list, metric_func=None, seed=42
    ) -> np.ndarray:
        key = (dataset_name, split, scaler_name, metric_name)
        
        # 1. Mirar en memoria
        if key in self._memory_cache:
            return self._memory_cache[key]
            
        # 2. Mirar en disco
        filename = f"dist_matrix_{dataset_name}_{split}_{scaler_name}_{metric_name}.npy"
        filepath = os.path.join(CACHE_DIR, filename)
        
        if os.path.exists(filepath):
            # logger.info(f"[{dataset_name}] Cargando matriz de distancias desde caché ({filename}).")
            matrix = np.load(filepath)
            self._memory_cache[key] = matrix
            return matrix
            
        # 3. Calcular y guardar
        if metric_func is None:
            raise ValueError("metric_func no fue proporcionado y la matriz no está en caché.")
            
        print(f"[{dataset_name}] Calculando matriz de distancias para {scaler_name} + {metric_name} ({split})...")
        matrix = compute_distance_matrix(bags, metric_func, metric_name)
        
        # Guardar en disco y memoria
        np.save(filepath, matrix)
        self._memory_cache[key] = matrix
        return matrix
        
    def clear_memory(self):
        self._memory_cache.clear()

# Instancia global
global_persistent_cache = PersistentDistanceMatrixCache()
