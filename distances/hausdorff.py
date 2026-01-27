import numpy as np
from scipy.spatial.distance import cdist
from data.bag import Bag

def hausdorff_distance(bag1: Bag, bag2: Bag) -> float:
    """
    Calcula la distancia de Hausdorff entre dos bolsas.
    Métrica: Distancia Euclidiana entre instancias.
    """
    # Obtenemos matrices numpy (n_inst x n_attr)
    mat1 = bag1.as_matrix()
    mat2 = bag2.as_matrix()
    
    if len(mat1) == 0 or len(mat2) == 0:
        return float('inf') # Manejo de bolsas vacías

    # Calculamos matriz de distancias cruzadas entre todas las instancias
    # Si bag1 tiene 5 instancias y bag2 tiene 10, d_matrix es 5x10
    d_matrix = cdist(mat1, mat2, metric='euclidean')
    
    # Calculamos Hausdorff dirigido h(A, B) y h(B, A)
    # min(axis=1): para cada fila (instancia de A), la dist mínima a B
    # max(...): la peor de esas distancias
    h_A_B = np.max(np.min(d_matrix, axis=1))
    
    # min(axis=0): para cada columna (instancia de B), la dist mínima a A
    h_B_A = np.max(np.min(d_matrix, axis=0))
    
    # Devolvemos el máximo de ambos
    return max(h_A_B, h_B_A)