import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
from miclustering.data.bag import Bag

def cauchy_schwarz_distance(bag1: Bag, bag2: Bag) -> float:
    """Calcula la distancia basada en Cauchy-Schwarz entre dos bolsas.
    
    La distancia de Cauchy-Schwarz se define como:
    d(A, B) = 1 - (⟨A, B⟩ / (||A|| * ||B||))
    
    donde ⟨A, B⟩ es el producto interno y ||·|| es la norma.
    
    Para bolsas, agregamos las instancias en un único vector representativo
    (por ejemplo, usando la media).

    Args:
        bag1: Primera bolsa.
        bag2: Segunda bolsa.

    Returns:
        Distancia de Cauchy-Schwarz entre las dos bolsas.
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

def earth_movers_distance(bag1: Bag, bag2: Bag) -> float:
    """Distancia Earth Mover's Distance (EMD) entre dos bolsas.
 
    Definición formal (ec. 3.27):
        D_EDM(A, B) = sum_{a in A, b in B} f(a,b) * d(a,b)
 
    donde d(a,b) es la distancia euclidiana entre instancias (ground distance)
    y f(a,b) es el flujo óptimo que minimiza el coste total de transporte,
    sujeto a las restricciones:
 
        1. f(a,b) >= 0
        2. sum_{a in A} f(a,b) <= 1/n_b   (no se extrae más de lo disponible en B)
        3. sum_{b in B} f(a,b) <= 1/n_a   (no se deposita más de lo disponible en A)
        4. sum_{a in A, b in B} f(a,b) = 1 (toda la tierra es transportada)
 
    Cada instancia tiene masa uniforme 1/n_a (bolsa A) y 1/n_b (bolsa B).
    El problema se formula como un Programa Lineal y se resuelve con scipy.

    Args:
        bag1: Primera bolsa.
        bag2: Segunda bolsa.

    Returns:
        Distancia EMD entre las dos bolsas.
    """
    mat1 = bag1.as_matrix()
    mat2 = bag2.as_matrix()
 
    if len(mat1) == 0 or len(mat2) == 0:
        return float('inf')
 
    n_a = len(mat1)
    n_b = len(mat2)
 
    # Ground distance: matriz euclidiana (n_a x n_b) 
    D = cdist(mat1, mat2, metric='euclidean')   # (n_a, n_b)
 
    # Formulación LP
    # Variable: f = vector plano (n_a * n_b,) con el flujo f(a_i, b_j)
    # Objetivo: minimizar sum_{i,j} D[i,j] * f[i,j]
    c = D.flatten()                             # (n_a*n_b,)
 
    n_vars = n_a * n_b
 
    # Restricción 4 (igualdad): sum_{i,j} f[i,j] = 1
    A_eq = np.ones((1, n_vars))
    b_eq = np.array([1.0])
 
    # Restricción 2: para cada b_j,  sum_{i} f[i,j] <= 1/n_b
    # Restricción 3: para cada a_i,  sum_{j} f[i,j] <= 1/n_a
    A_ub_rows = []
    b_ub_vals = []
 
    # sum_{i} f[i,j] <= 1/n_b  para cada j en [0, n_b)
    for j in range(n_b):
        row = np.zeros(n_vars)
        for i in range(n_a):
            row[i * n_b + j] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(1.0 / n_b)
 
    # sum_{j} f[i,j] <= 1/n_a  para cada i en [0, n_a)
    for i in range(n_a):
        row = np.zeros(n_vars)
        row[i * n_b : (i + 1) * n_b] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(1.0 / n_a)
 
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
 
    # Restricción 1: f(a,b) >= 0  → bounds (0, None) por defecto en linprog
    bounds = [(0, None)] * n_vars
 
    result = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method='highs',       # solver moderno y robusto de scipy
    )
 
    if not result.success:
        # Fallback: si el LP no converge, devolvemos la distancia entre centroides
        return float(np.linalg.norm(np.mean(mat1, axis=0) - np.mean(mat2, axis=0)))
 
    return float(result.fun)

def mahalanobis_distance(bag1: Bag, bag2: Bag) -> float:
    """Distancia de Mahalanobis entre dos bolsas.
 
    Definición formal (ec. 3.28):
        D_Mahalanobis(A, B) = (μ_a - μ_b)^T * (½Σ_a + ½Σ_b)^{-1} * (μ_a - μ_b)
 
    donde:
        μ_a, μ_b  = medias de las instancias de las bolsas A y B
        Σ_a, Σ_b  = matrices de covarianza de las instancias de A y B
 
    La matriz combinada (½Σ_a + ½Σ_b) es la covarianza promediada de ambas
    distribuciones gaussianas que aproximan las bolsas.

    Args:
        bag1: Primera bolsa.
        bag2: Segunda bolsa.

    Returns:
        Distancia de Mahalanobis entre las dos bolsas.
    """
    mat1 = bag1.as_matrix()
    mat2 = bag2.as_matrix()
 
    if len(mat1) == 0 or len(mat2) == 0:
        return float('inf')
 
    #  Medias 
    mu_a = np.mean(mat1, axis=0)   # (d,)
    mu_b = np.mean(mat2, axis=0)   # (d,)
    diff = mu_a - mu_b             # (d,)
 
    #  Covarianzas (rowvar=False: cada columna es una variable) 
    # np.cov necesita al menos 2 instancias para estimar covarianza.
    # Con 1 instancia la covarianza es cero → usamos identidad como fallback.
    if len(mat1) < 2:
        cov_a = np.eye(mat1.shape[1])
    else:
        cov_a = np.cov(mat1, rowvar=False)   # (d, d)
 
    if len(mat2) < 2:
        cov_b = np.eye(mat2.shape[1])
    else:
        cov_b = np.cov(mat2, rowvar=False)   # (d, d)
 
    #  Covarianza combinada: ½Σ_a + ½Σ_b 
    cov_combined = 0.5 * cov_a + 0.5 * cov_b   # (d, d)

    epsilon = 1e-5
    cov_combined += np.eye(cov_combined.shape[0]) * epsilon
 
    #  Inversión con regularización si es necesario 
    # Intentamos inversión directa; si falla (singular) usamos pseudoinversa.
    try:
        cov_inv = np.linalg.inv(cov_combined)
    except np.linalg.LinAlgError:
        try:
            cov_inv = np.linalg.pinv(cov_combined)
        except np.linalg.LinAlgError:
            # Fallback extremo: si SVD tampoco converge, usamos la matriz identidad.
            # Esto convierte efectivamente a Mahalanobis en una distancia Euclidiana
            # para este par de bolsas concreto, evitando que el script colapse.
            cov_inv = np.eye(cov_combined.shape[0])
 
    # Comprobación adicional: si la inversa tiene valores muy grandes
    # (casi-singularidad), también usamos pseudoinversa.
    if not np.all(np.isfinite(cov_inv)):
        cov_inv = np.linalg.pinv(cov_combined)
 
    #  Distancia de Mahalanobis 
    # d = (μ_a - μ_b)^T * Σ^{-1} * (μ_a - μ_b)
    # Devolvemos la raíz cuadrada para que la escala sea comparable a
    # distancias euclidianas (mismas unidades que el espacio original).
    maha_sq = float(diff @ cov_inv @ diff)
 
    # Protección numérica: por errores de punto flotante puede ser ligeramente negativo
    return float(np.sqrt(max(0.0, maha_sq)))