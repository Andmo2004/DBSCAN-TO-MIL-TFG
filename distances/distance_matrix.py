import logging
import numpy as np
from typing import List, Callable

from data.bag import Bag

logger = logging.getLogger(__name__)


def compute_distance_matrix(bags: List[Bag], 
                           metric_func: Callable[[Bag, Bag], float], 
                           metric_name: str = "custom") -> np.ndarray:
    """
    Calcula la matriz de distancias simétrica usando la métrica especificada.
    
    :param bags: (List[Bag]) Lista de Bolsas
    :param metric_func: (Callable) Función de distancia que acepta dos Bags
    :param metric_name: (str) Nombre de la métrica (solo para logging)
    :return: (ndarray) Matriz numpy de distancias (N X N)
    """
    
    num_bags = len(bags)
    logger.info(f"Calculando matriz ({num_bags}x{num_bags}) usando métrica: '{metric_name}'...")

    # Inicializamos matriz a 0
    matrix = np.zeros((num_bags, num_bags))

    for i in range(num_bags):
        bag_a = bags[i]
        for j in range(i + 1, num_bags):
            bag_b = bags[j]
            
            d = metric_func(bag_a, bag_b)

            matrix[i, j] = d
            matrix[j, i] = d

    logger.debug("Cálculo de matriz de distancias finalizado.")
    logger.debug(f"Matriz:\n{matrix}")
    return matrix
