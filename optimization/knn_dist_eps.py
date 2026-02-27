import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, List

######## SOLO PARA TEST UNITARIO
import os
import sys
# Agregar root del proyecto al path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
#########

from data.bag import Bag
from data.midata import MIData
from distances.distance_matrix import compute_distance_matrix

logger = logging.getLogger(__name__)


def _find_knee_point(distances: np.ndarray) -> Tuple[int, float]:
    """
    Encuentra el 'codo' de la curva usando la máxima distancia perpendicular
    desde cada punto hasta la línea secante que une el primero y el último.

    Ambos ejes se normalizan a [0, 1] antes del cálculo para evitar el sesgo
    que produce la diferencia de escala entre índices (eje X) y distancias (eje Y).
    """
    n = len(distances)
    if n == 0:
        return 0, 0.0

    # Normalizar ejes a [0, 1] para que la distancia perpendicular sea imparcial
    x = np.linspace(0.0, 1.0, n)
    y_min, y_max = distances.min(), distances.max()
    y = (distances - y_min) / (y_max - y_min + 1e-10)

    p1 = np.array([x[0],  y[0]])
    p2 = np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    max_dist = -1.0
    knee_idx = 0

    for i in range(n):
        p3 = np.array([x[i], y[i]])
        cross = np.abs(np.cross(line_vec, p1 - p3))
        perp_dist = cross / (line_len + 1e-10)

        if perp_dist > max_dist:
            max_dist = perp_dist
            knee_idx = i

    return knee_idx, float(distances[knee_idx])


def optimize_eps(
    dataset: MIData,
    distance_func: Callable[[Bag, Bag], float],
    min_pts: Optional[int] = None,
    try_range: bool = False,
    save_plots: bool = True,
) -> float:
    """
    Determina el valor óptimo de eps mediante el k-NN distance plot.

    Recibe el dataset YA normalizado/estandarizado: no aplica ningún escalado
    internamente. El escalado debe realizarse antes de llamar a esta función.

    :param dataset:       MIData ya escalado (MinMax o Standard).
    :param distance_func: Función (Bag, Bag) -> float.
    :param min_pts:       minPts de DBSCAN. Si None, se usa la heurística 2*d.
    :param try_range:     Si True, explora k de 1 a 20 en lugar de solo k = minPts-1.
    :param save_plots:    Si True, guarda los gráficos en output_dir (no los muestra).
    :param output_dir:    Directorio donde se guardan los PNGs.
    :return:              Valor de eps estimado en el codo más pronunciado.
    """
    import os

    output_dir: str = f"results/output_eps/{dataset._name}"
    os.makedirs(output_dir, exist_ok=True)

    # Workflow propuesto en (https://deepwiki.com/mhahsler/dbscan/10.1-selecting-eps-for-dbscan)

    # Detectamos dimensionalidad desde la primera instancia de la primera bolsa
    first_bag = dataset.get_bag(0)
    d = first_bag.get_instance(0).num_attributes()
    logger.info(f"Dimensionalidad detectada: d={d}")

    # Elegimos el parámetro minPts
    if min_pts is None:
        min_pts = 2 * d
        logger.info(f"minPts no proporcionado. Usando heurística 2*d = {min_pts}")

    # Establecemos el rango de k a explorar
    k_values = list(range(1, 21)) if try_range else [min_pts - 1]
    if try_range:
        logger.info("Modo try_range activado: explorando k = 1..20")

    # Calculamos la matriz de distancias una sola vez (dataset ya escalado)
    dist_matrix = compute_distance_matrix(dataset.bags, distance_func, metric_name="k-NN")

    best_eps   = 0.0
    best_knee  = -1.0   # máxima distancia perpendicular encontrada

    n_bags = dataset.get_num_bags()

    for k in k_values:
        if k >= n_bags:
            logger.warning(f"k={k} >= n_bags={n_bags}. Saltando...")
            continue

        # Calculamos distancia al k-ésimo vecino real (excluimos distancia consigo mismo)
        k_distances = []
        for row in dist_matrix:
            neighbors = np.sort(row[row > 0])   # excluye diagonal (dist. a sí mismo)
            if len(neighbors) >= k:
                k_distances.append(neighbors[k - 1])
            else:
                k_distances.append(neighbors[-1])   # fallback: vecino más lejano disponible

        k_distances = np.sort(np.array(k_distances))

        # Buscamos el codo
        knee_idx, eps_candidate = _find_knee_point(k_distances)

        # Calculamos la "prominencia" del codo en escala normalizada para comparar entre k
        x = np.linspace(0.0, 1.0, len(k_distances))
        y_min, y_max = k_distances.min(), k_distances.max()
        y = (k_distances - y_min) / (y_max - y_min + 1e-10)
        p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
        line_vec = p2 - p1
        p3 = np.array([x[knee_idx], y[knee_idx]])
        prominence = np.abs(np.cross(line_vec, p1 - p3)) / (np.linalg.norm(line_vec) + 1e-10)

        if prominence > best_knee:
            best_knee = prominence
            best_eps  = eps_candidate

        logger.info(f"k={k:>3} | knee_idx={knee_idx:>4} | eps≈{eps_candidate:.6f} | prominencia={prominence:.6f}")

        # Visualización: guarda PNG.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_distances, linewidth=2, label=f'{k}-NN Distances')
        ax.axhline(y=eps_candidate, color='r', linestyle='--',
                   label=f'Knee  eps ≈ {eps_candidate:.4f}')
        ax.axvline(x=knee_idx, color='r', linestyle=':', alpha=0.5)
        ax.scatter(knee_idx, eps_candidate, color='red', s=60, zorder=5)
        ax.set_title(f'k-NN Distance Plot  (k={k})')
        ax.set_xlabel('Bolsas ordenadas por distancia al k-ésimo vecino')
        ax.set_ylabel(f'Distancia al {k}-ésimo vecino más cercano')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_plots:
            path = os.path.join(output_dir, f"knn_dist_k{k:02d}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Gráfico guardado → {path}")
        else:
            plt.show()

        plt.close(fig)

    ## logger.info(f"Eps óptimo seleccionado: {best_eps:.6f}")
    
    # Sanity check: si eps > percentil 50 de distancias, probablemente es demasiado grande
    upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    upper = upper[upper > 0]
    p50 = float(np.percentile(upper, 50))
    if best_eps > p50:
        logger.warning(f"Eps óptimo ({best_eps:.4f}) > percentil 50 ({p50:.4f}). Usando percentil 20.")
        best_eps = float(np.percentile(upper, 20))

    return best_eps


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from preprocessing.scaler import MinMaxScaler
    from distances.hausdorff import hausdorff_distance
    
    try:
        # 1. Cargar dataset
        logger.info("Cargando dataset musk1.arff...")
        dataset = MIData.from_arff("datasets/musk1.arff")
        logger.info(f"Dataset cargado: {dataset.get_num_bags()} bolsas")
        
        # 2. Normalizar con MinMaxScaler
        logger.info("Normalizando dataset con MinMaxScaler...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_normalized = scaler.fit_transform(dataset)
        logger.info(f"Dataset normalizado correctamente")
        
        # 3. Optimizar eps
        logger.info("Optimizando eps para DBSCAN...")
        optimal_eps = optimize_eps(
            dataset=dataset_normalized,
            distance_func=hausdorff_distance,
            min_pts=None,  # Usa heurística 2*d
            try_range=True,  # Explora k=1..20
            save_plots=True,
        )
        
        logger.info(f"\nEps óptimo encontrado: {optimal_eps:.6f}")
        
    except Exception as e:
        logger.error(f"Error en el test: {e}", exc_info=True)