import os
import numpy as np
import logging
from typing import Dict, Optional, Any, Callable
from collections import OrderedDict
from miclustering.distances.distance_matrix import compute_distance_matrix

logger = logging.getLogger(__name__)

# La librería usa un directorio local por defecto, pero permite sobreescribirlo
DEFAULT_CACHE_DIR = os.path.join(os.getcwd(), ".miclustering_cache", "distance_matrices")

def _get_cache_dir():
    """Lee la variable de entorno en tiempo de ejecución para permitir cambios dinámicos."""
    return os.environ.get("MICLUSTERING_CACHE_DIR", DEFAULT_CACHE_DIR)

class PersistentDistanceMatrixCache:
    """Caché LRU de matrices de distancias (disco + memoria).
 
    El diccionario en memoria está limitado a ``maxsize`` entradas.
    Cuando se supera ese límite se expulsa la entrada usada hace más tiempo
    (política LRU), liberando la RAM que ocupa esa matriz N×N.
    """
    
    def __init__(self, maxsize: int = 3) -> None:
        """
        Args:
            maxsize: Número máximo de matrices a mantener en RAM simultáneamente.
                     Con matrices de ~400 bolsas * 400 bolsas * float64 ≈ 1,3 MB;
                     3 matrices = ~4 MB de caché en memoria.
                     Sube a 6-8 si tienes ≥16 GB de RAM disponibles en Kaggle.
        """
        cache_dir = _get_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        self._maxsize = maxsize
        # OrderedDict preserva el orden de uso: move_to_end() sube al frente.
        self._memory_cache: OrderedDict = OrderedDict()

    def get(
        self,
        dataset_name: str,
        split: str,
        scaler_name: str,
        metric_name: str,
        bags: list,
        metric_func: Optional[Callable] = None,
        seed: int = 42,
        save: bool = False,
        n_jobs: int = 1,
        device: str = "cpu",
    ) -> np.ndarray:
        key = (dataset_name, split, scaler_name, metric_name)
 
        # 1. Memoria (LRU hit → mover al frente para marcar como "recién usado")
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            return self._memory_cache[key]
 
        # 2. Disco
        filename = (
            f"dist_matrix__{dataset_name}__{split}"
            f"__{scaler_name}__{metric_name}.npy"
        )
        filepath = os.path.join(_get_cache_dir(), filename)
 
        if os.path.exists(filepath):
            matrix = np.load(filepath)
            expected = (len(bags), len(bags))
            if matrix.shape != expected:
                logger.warning(
                    f"Matriz en caché {filepath} tiene shape {matrix.shape}, "
                    f"se esperaba {expected}. Recalculando."
                )
            else:
                self._lru_set(key, matrix)
                return matrix
 
        # 3. Calcular
        if metric_func is None:
            raise ValueError(
                "metric_func no fue proporcionado y la matriz no está en caché."
            )
        from miclustering.distances.distance_matrix import compute_distance_matrix  # noqa: PLC0415
        logger.info(
            f"[{dataset_name}] Calculando matriz "
            f"({split} / {scaler_name} / {metric_name}) [device={device}, n_jobs={n_jobs}]..."
        )
        matrix = compute_distance_matrix(
            bags, metric_func, metric_name, n_jobs=n_jobs, device=device
        )
 
        self._lru_set(key, matrix)
        if save:
            np.save(filepath, matrix)
            logger.debug(f"Matriz guardada en disco → {filepath}")
        else:
            logger.debug("Matriz calculada pero NO persistida (save=False)")
 
        return matrix
        
    def clear_memory(self):
        self._memory_cache.clear()

    def _lru_set(self, key: tuple, matrix: np.ndarray) -> None:
        """Inserta una entrada en la caché LRU, expulsando la más antigua si es necesario."""
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
        else:
            if len(self._memory_cache) >= self._maxsize:
                # popitem(last=False) elimina el elemento más antiguo (FIFO del OrderedDict)
                evicted_key, _ = self._memory_cache.popitem(last=False)
                logger.debug(f"[LRU] Expulsada de memoria: {evicted_key}")
            self._memory_cache[key] = matrix

# Instancia global
global_persistent_cache = PersistentDistanceMatrixCache(maxsize=3)
