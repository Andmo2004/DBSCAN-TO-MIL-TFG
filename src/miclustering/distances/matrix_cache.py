import os
import numpy as np
import logging
from typing import Dict, Optional, Any, Callable
from miclustering.distances.distance_matrix import compute_distance_matrix

logger = logging.getLogger(__name__)

# La librería usa un directorio local por defecto, pero permite sobreescribirlo
DEFAULT_CACHE_DIR = os.path.join(os.getcwd(), ".miclustering_cache", "distance_matrices")
CACHE_DIR = os.environ.get("MICLUSTERING_CACHE_DIR", DEFAULT_CACHE_DIR)

class PersistentDistanceMatrixCache:
    """Caché para almacenar matrices de distancias en disco y evitar recalcularlas."""
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._memory_cache = {}

    def get(
        self,
        dataset_name: str,
        split: str,
        scaler_name: str,
        metric_name: str,
        bags: list,
        metric_func=None,
        seed: int = 42,
        save: bool = False
    ) -> np.ndarray:
        key = (dataset_name, split, scaler_name, metric_name)

        # 1. Memoria
        if key in self._memory_cache:
            return self._memory_cache[key]

        # 2. Disco
        filename = f"dist_matrix_{dataset_name}_{split}_{scaler_name}_{metric_name}.npy"
        filepath = os.path.join(CACHE_DIR, filename)

        if os.path.exists(filepath):
            matrix = np.load(filepath)
            self._memory_cache[key] = matrix
            return matrix

        # 3. Calcular
        if metric_func is None:
            raise ValueError("metric_func no fue proporcionado y la matriz no está en caché.")

        print(f"[{dataset_name}] Calculando matriz ({split} / {scaler_name} / {metric_name})...")
        matrix = compute_distance_matrix(bags, metric_func, metric_name)

        # 4. Guardar solo si save=True
        self._memory_cache[key] = matrix
        if save:
            np.save(filepath, matrix)
            logger.debug(f"Matriz guardada en disco → {filepath}")
        else:
            logger.debug(f"Matriz calculada pero NO persistida (save=False)")

        return matrix
        
    def clear_memory(self):
        self._memory_cache.clear()

# Instancia global
global_persistent_cache = PersistentDistanceMatrixCache()
