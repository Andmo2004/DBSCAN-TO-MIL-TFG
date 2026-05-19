import numpy as np
from scipy.spatial.distance import cdist
from miclustering.data.bag import Bag

#_________________

# def hausdorff_distance(bag1: Bag, bag2: Bag) -> float:
#     """
#     Calcula la distancia de Hausdorff entre dos bolsas.
#     Métrica: Distancia Euclidiana entre instancias.
#     """
#     # Obtenemos matrices numpy (n_inst x n_attr)
#     mat1 = bag1.as_matrix()
#     mat2 = bag2.as_matrix()
    
#     if len(mat1) == 0 or len(mat2) == 0:
#         return float('inf') # Manejo de bolsas vacías

#     # Calculamos matriz de distancias cruzadas entre todas las instancias
#     # Si bag1 tiene 5 instancias y bag2 tiene 10, d_matrix es 5x10
#     d_matrix = cdist(mat1, mat2, metric='euclidean')
    
#     # Calculamos Hausdorff dirigido h(A, B) y h(B, A)
#     # min(axis=1): para cada fila (instancia de A), la dist mínima a B
#     # max(...): la peor de esas distancias
#     h_A_B = np.max(np.min(d_matrix, axis=1))
    
#     # min(axis=0): para cada columna (instancia de B), la dist mínima a A
#     h_B_A = np.max(np.min(d_matrix, axis=0))
    
#     # Devolvemos el máximo de ambos
#     return max(h_A_B, h_B_A)

def _distance_matrix(bag1: Bag, bag2: Bag):
    """
    Calcula la matriz de distancias euclidianas entre las instancias de dos bolsas.
    Devuelve (mat1, mat2, d_matrix) 
    o (None, None, None) si alguna bolsa está vacía.
    """
    mat1 = bag1.as_matrix()
    mat2 = bag2.as_matrix()
    if len(mat1) == 0 or len(mat2) == 0:
        return None, None, None
    return mat1, mat2, cdist(mat1, mat2, metric='euclidean')

def hausdorff_distance(bag1: Bag, bag2: Bag) -> float:
    """
    Distancia de Hausdorff MÁXIMA (simétrica) entre dos bolsas.
 
    Definición formal (ec. 3.19 - 3.20):
        D_Hausdorff-max(A, B) = max{ h(A,B), h(B,A) }
        h(A,B) = max_{a in A} min_{b in B} d(a,b)
 
    Para cada instancia de A se busca su vecino más cercano en B (min_{b in B}).
    Se toma el peor caso (max_{a in A}).
    La distancia simétrica toma el máximo de ambas direcciones.
 
    """
    _, _, d_matrix = _distance_matrix(bag1, bag2)
    if d_matrix is None:
        return float('inf')
 
    # h(A,B): para cada fila (instancia de A), distancia mínima a B -> peor caso
    h_A_B = float(np.max(np.min(d_matrix, axis=1)))
    # h(B,A): para cada columna (instancia de B), distancia mínima a A -> peor caso
    h_B_A = float(np.max(np.min(d_matrix, axis=0)))
 
    return max(h_A_B, h_B_A)

def hausdorff_distance_min(bag1: Bag, bag2: Bag) -> float:
    """
    Distancia de Hausdorff mínima entre dos bolsas.
 
    Definición formal (ec. 3.18):
        D_Hausdorff-min(A, B) = min_{a in A} min_{b in B} d(a,b)
 
    Mínimo absoluto de la matriz de distancias cruzadas.
 
    """
    _, _, d_matrix = _distance_matrix(bag1, bag2)
    if d_matrix is None:
        return float('inf')
 
    # Mínimo absoluto de toda la matriz de distancias cruzadas
    return float(np.min(d_matrix))


def hausdorff_distance_avg(bag1: Bag, bag2: Bag) -> float:
    """
    Distancia de Hausdorff PROMEDIO entre dos bolsas.
 
    Definición formal (ec. 3.21):
 
        D_Hausdorff-avg(A, B) = [ sum_{a in A} min_{b in B} d(a,b)
                                 + sum_{b in B} min_{a in A} d(b,a) ]
                                 / (|A| + |B|)
 
    Para cada instancia de A se busca su vecino más cercano en B y se acumula.
    Ídem para cada instancia de B hacia A.
    El resultado se normaliza por el número total de instancias de ambas bolsas.

    """
    mat1, mat2, d_matrix = _distance_matrix(bag1, bag2)
    if d_matrix is None:
        return float('inf')
 
    # sum_{a in A} min_{b in B} d(a,b): por cada fila (instancia A), min hacia B
    sum_A_to_B = float(np.sum(np.min(d_matrix, axis=1)))
 
    # sum_{b in B} min_{a in A} d(b,a): por cada columna (instancia B), min hacia A
    sum_B_to_A = float(np.sum(np.min(d_matrix, axis=0)))
 
    assert mat1 is not None and mat2 is not None

    total_instances = len(mat1) + len(mat2)
 
    return (sum_A_to_B + sum_B_to_A) / total_instances