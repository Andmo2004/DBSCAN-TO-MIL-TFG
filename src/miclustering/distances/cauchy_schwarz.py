import numpy as np
from miclustering.data.bag import Bag

def cauchy_schwarz_distance(bag1: Bag, bag2: Bag) -> float:
    """
    Calcula la distancia basada en Cauchy-Schwarz entre dos bolsas.
    
    La distancia de Cauchy-Schwarz se define como:
    d(A, B) = 1 - (⟨A, B⟩ / (||A|| * ||B||))
    
    donde ⟨A, B⟩ es el producto interno y ||·|| es la norma.
    
    Para bolsas, agregamos las instancias en un único vector representativo
    (por ejemplo, usando la media).
    """
    # Obtenemos matrices numpy
    mat1 = bag1.as_matrix()
    mat2 = bag2.as_matrix()
    
    if len(mat1) == 0 or len(mat2) == 0:
        return float('inf')  # Manejo de bolsas vacías
    
    # Agregamos las instancias de cada bolsa en un vector representativo
    # Media de las instancias
    vec1 = np.mean(mat1, axis=0)
    vec2 = np.mean(mat2, axis=0)
    
    # Calculamos producto interno
    dot_product = np.dot(vec1, vec2)
    
    # Calculamos normas
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Evitamos división por cero
    if norm1 == 0 or norm2 == 0:
        return float('inf')
    
    # Calculamos similitud coseno (Cauchy-Schwarz normalizado)
    cos_similarity = dot_product / (norm1 * norm2)
    
    # Convertimos similitud a distancia
    # Aseguramos que cos_similarity esté en [-1, 1] por errores numéricos
    cos_similarity = np.clip(cos_similarity, -1.0, 1.0)
    
    # Distancia: 1 - similitud (rango [0, 2])
    distance = 1.0 - cos_similarity
    
    return float(distance)