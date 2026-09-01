import logging
import numpy as np
from typing import List, Callable

from miclustering.data.bag import Bag

logger = logging.getLogger(__name__)


def compute_distance_matrix(
    bags: List[Bag], 
    metric_func: Callable[[Bag, Bag], float], 
    metric_name: str = "custom",
    n_jobs: int = 1,
    device: str = "cpu",
) -> np.ndarray:
    """Calcula la matriz de distancias simétrica usando la métrica especificada.
    
    Args:
        bags: Lista de Bolsas.
        metric_func: Función de distancia que acepta dos Bags.
        metric_name: Nombre de la métrica (solo para logging).
        n_jobs: Número de procesos paralelos para el cómputo (-1 para todos los núcleos).
                Si es 1, ejecuta la versión secuencial sin sobrecarga de procesos.
        device: Dispositivo para aceleración ('cpu', 'cuda', 'mps', 'auto').
                Si es distinto de 'cpu' y PyTorch está disponible, utiliza el backend GPU.

    Returns:
        Matriz numpy simétrica de distancias (N X N).
    """
    
    num_bags = len(bags)
    if num_bags <= 1:
        return np.zeros((num_bags, num_bags), dtype=np.float64)

    # Intentar aceleración por GPU si device != "cpu"
    if device and device.lower() != "cpu":
        try:
            from miclustering.distances.torch_backend import is_torch_available, get_torch_device, compute_distance_matrix_torch
            if is_torch_available():
                resolved_dev = get_torch_device(device)
                if resolved_dev is not None and resolved_dev.type != "cpu":
                    return compute_distance_matrix_torch(bags, metric_name=metric_name, device=device)
                # Si resolved_dev es CPU (p. ej. GPU incompatible como Tesla P100 o sin GPU), usar CPU multinúcleo
            else:
                logger.warning("Dispositivo GPU solicitado pero PyTorch no está disponible. Degradando a CPU.")
        except Exception as e:
            logger.warning(f"Error en cómputo GPU ({e}). Degradando a CPU.")

    logger.info(f"Calculando matriz ({num_bags}x{num_bags}) usando métrica: '{metric_name}' (n_jobs={n_jobs}, device=cpu)...")

    # Caso especial: Mahalanobis con estadísticos precomputados — O(N) covarianzas
    # en vez de O(N²). Resultado numérico idéntico al bucle genérico.
    if metric_name.lower() == "mahalanobis":
        from miclustering.distances.probability_distribution import compute_mahalanobis_matrix  # noqa: PLC0415
        return compute_mahalanobis_matrix(bags)

    # Inicializamos matriz a 0
    matrix = np.zeros((num_bags, num_bags), dtype=np.float64)

    # Si n_jobs == 1 o el tamaño es muy pequeño, evitamos sobrecarga de procesos
    if n_jobs == 1 or num_bags <= 15:
        for i in range(num_bags):
            bag_a = bags[i]
            for j in range(i + 1, num_bags):
                d = metric_func(bag_a, bags[j])
                matrix[i, j] = d
                matrix[j, i] = d
    else:
        from joblib import Parallel, delayed

        def _compute_row(i: int):
            bag_a = bags[i]
            return i, [metric_func(bag_a, bags[j]) for j in range(i + 1, num_bags)]

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_compute_row)(i) for i in range(num_bags - 1)
        )
        for i, row_vals in results:
            for offset, d in enumerate(row_vals):
                j = i + 1 + offset
                matrix[i, j] = d
                matrix[j, i] = d

    logger.debug("Cálculo de matriz de distancias finalizado.")
    logger.debug(f"Matriz:\n{matrix}")
    return matrix
