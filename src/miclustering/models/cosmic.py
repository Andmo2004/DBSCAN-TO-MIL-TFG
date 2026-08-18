"""
Implementación del algoritmo COSMIC (COnceptual Specified Multi-Instance Clusters)
para Multi-Instance Learning, adaptado de OPTICS.

Referencia:
    Kriegel, H. P., Pryakhin, A., Schubert, M., Zimek, A. (2006). COSMIC: conceptually
    specified multi-instance clusters. Proceedings of the 6th International Conference
    on Data Mining (ICDM 2006), pp. 917-921.

NOTA IMPORTANTE:
Esta clase implementa el Paso 1 (ordenamiento por densidad, Algoritmo 5) y el Paso 2
(extracción de clusters mediante epsilon', Algoritmo 6) del artículo original. La
generación de "conceptos" (la mitad del nombre COSMIC, basada en Formal Concept
Analysis sobre atributos de las instancias) NO está implementada: el capítulo de
referencia no detalla su procedimiento exacto y depende de una definición de atributos
por instancia que queda fuera del alcance de esta clase. Si se retoma en el futuro,
encajaría como un método adicional (p. ej. `derive_concepts()`).
"""

import heapq
import logging
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

# NOTA: ajusta estas importaciones a la estructura real de tu paquete MIClustering.
# Se asume la misma organización de módulos que usa MIDBSCAN.py (Bag, MIData,
# DISTANCE_REGISTRY, compute_distance_matrix, logger).

from miclustering.data.midata import MIData
from miclustering.data.bag import Bag
from miclustering.distances import DISTANCE_REGISTRY
from miclustering.distances.distance_matrix import compute_distance_matrix

logger = logging.getLogger(__name__)


class COSMIC(BaseEstimator, ClusterMixin):
    """
    Implementación del algoritmo COSMIC (density-based, basado en OPTICS) adaptado
    para Multi-Instance Learning.

    A diferencia de MIDBSCAN (adaptación directa de DBSCAN), COSMIC construye primero
    un ordenamiento de las bolsas por densidad (independiente de un único punto de
    corte), y después extrae particiones concretas usando un umbral epsilon' <= epsilon,
    sin necesidad de recalcular el ordenamiento. Esto permite explorar varias
    granularidades de clustering a partir de un único ajuste.
    """

    NOISE_LABEL = -1

    def __init__(
        self,
        epsilon: float,
        min_pts: int,
        epsilon_prime: Optional[float] = None,
        metric: str = "hausdorff",
    ):
        """Constructor del modelo COSMIC.

        Args:
            epsilon: Radio máximo usado para construir el ordenamiento (vecindad de referencia).
            min_pts: Número mínimo de bolsas en la vecindad para considerar un objeto núcleo.
            epsilon_prime: Umbral de extracción de clusters (epsilon_prime <= epsilon). Si no
                se proporciona, se usa epsilon en la primera extracción automática dentro de fit().
            metric: Métrica de distancia entre bolsas a usar.

        Raises:
            ValueError: Si epsilon <= 0, min_pts < 1, o epsilon_prime > epsilon.
        """
        if epsilon <= 0:
            raise ValueError(f"El parámetro 'epsilon' debe ser > 0. Recibido: {epsilon}")
        if min_pts < 1:
            raise ValueError(f"El parámetro 'min_pts' debe ser >= 1. Recibido: {min_pts}")
        if epsilon_prime is not None and epsilon_prime > epsilon:
            raise ValueError(
                f"'epsilon_prime' ({epsilon_prime}) no puede ser mayor que 'epsilon' ({epsilon})."
            )

        # Parámetros del algoritmo
        self._epsilon = epsilon
        self._min_pts = min_pts
        self._epsilon_prime = epsilon_prime
        self._metric_name = metric.lower()
        self._metric_func = self._get_metric_function(self._metric_name)

        # Estado del modelo
        self._fitted = False
        self._train_bags: List[Bag] = []

        # Resultado del Paso 1 (ordenamiento OPTICS), indexado por posición en _train_bags
        self._ordering: Optional[List[int]] = None
        self._core_distance: List[Optional[float]] = []
        self._reachability_distance: List[Optional[float]] = []

        # Resultado del Paso 2 (extracción de clusters)
        self._labels: Dict[str, int] = {}
        self._cluster_count = 0
        self._core_bags: List[Bag] = []
        self._core_bag_labels: Dict[str, int] = {}

        logger.debug(
            f"COSMIC inicializado: epsilon={epsilon}, min_pts={min_pts}, "
            f"epsilon_prime={epsilon_prime}"
        )

    # Propiedades @property (solo lectura)
    @property
    def epsilon(self) -> float:
        """Radio usado para construir el ordenamiento (solo lectura)."""
        return self._epsilon

    @property
    def min_pts(self) -> int:
        """Mínimo de bolsas para considerar un objeto núcleo (solo lectura)."""
        return self._min_pts

    @property
    def epsilon_prime(self) -> Optional[float]:
        """Umbral usado en la última extracción de clusters (solo lectura)."""
        return self._epsilon_prime

    @property
    def cluster_count(self) -> int:
        """Número de clusters encontrados en la última extracción (excluyendo ruido)."""
        return self._cluster_count

    @property
    def labels(self) -> Dict[str, int]:
        """Devuelve un diccionario con las etiquetas de la última extracción {bag_id: cluster_id}."""
        return self._labels.copy()

    @property
    def noise_label(self) -> int:
        """Etiqueta utilizada para bolsas de ruido."""
        return self.NOISE_LABEL

    @property
    def is_fitted(self) -> bool:
        """Indica si el ordenamiento (Paso 1) ya fue calculado."""
        return self._fitted

    @property
    def ordering(self) -> List[str]:
        """Devuelve el ordenamiento de bag_ids obtenido en el Paso 1 (reachability plot)."""
        if self._ordering is None:
            return []
        return [self._train_bags[idx].bag_id for idx in self._ordering]

    @property
    def reachability_plot(self) -> List[Optional[float]]:
        """
        Devuelve la lista de reachability-distances en el orden de `ordering`, útil
        para graficar el reachability plot característico de OPTICS/COSMIC.
        """
        if self._ordering is None:
            return []
        return [self._reachability_distance[idx] for idx in self._ordering]

    # Método _reset_state()
    def _reset_state(self):
        """Reinicia el estado interno del modelo antes de un nuevo ajuste."""
        self._fitted = False
        self._train_bags = []
        self._ordering = None
        self._core_distance = []
        self._reachability_distance = []
        self._labels = {}
        self._cluster_count = 0
        self._core_bags = []
        self._core_bag_labels = {}

    def _get_metric_function(self, name: str) -> Callable[[Bag, Bag], float]:
        """
        Selecciona la función de distancia basada en el nombre usando el registro central.
        """
        if name not in DISTANCE_REGISTRY:
            valid_keys = list(DISTANCE_REGISTRY.keys())
            raise ValueError(f"Métrica '{name}' no reconocida. Disponibles: {valid_keys}")
        return DISTANCE_REGISTRY[name]

    def _compute_distance_matrix(self, bags: List[Bag]) -> np.ndarray:
        """Calcula la matriz de distancias usando el módulo externo.

        Args:
            bags: Lista de bolsas.

        Returns:
            Matriz numpy de distancias (N x N).
        """
        return compute_distance_matrix(bags, self._metric_func, self._metric_name)

    def _add_core_point(self, bag: Bag, cluster_id: int):
        """Registra un punto como núcleo, asociado al cluster en el que fue encontrado."""
        self._core_bags.append(bag)
        self._core_bag_labels[bag.bag_id] = cluster_id

    def _core_distance_of(self, idx: int, dist_matrix: np.ndarray) -> Optional[float]:
        """
        Calcula la core-distance de la bolsa en `idx`: la distancia al min_pts-ésimo
        vecino más cercano (incluyéndose a sí misma) dentro de su epsilon-vecindad.

        Returns:
            La core-distance, o None si la bolsa no es un objeto núcleo (menos de
            min_pts vecinos dentro de epsilon).
        """
        distances = dist_matrix[idx]
        neighbor_distances = distances[distances <= self._epsilon]
        if len(neighbor_distances) < self._min_pts:
            return None
        return float(np.sort(neighbor_distances)[self._min_pts - 1])

    def _update_seeds(
        self,
        core_idx: int,
        core_dist: float,
        dist_matrix: np.ndarray,
        visited: np.ndarray,
        seed_heap: list,
    ):
        """
        Actualiza las reachability-distances de los vecinos no visitados de un objeto
        núcleo e inserta/actualiza sus entradas en la cola de prioridad. Equivale a la
        ControlList del Algoritmo 5, implementada aquí con un heap + borrado perezoso
        (heapq no soporta decrease-key nativo).
        """
        neighbors_index = np.where(dist_matrix[core_idx] <= self._epsilon)[0]
        for neighbor_idx in neighbors_index:
            if visited[neighbor_idx]:
                continue
            new_reach = max(core_dist, float(dist_matrix[core_idx, neighbor_idx]))
            current_reach = self._reachability_distance[neighbor_idx]
            if current_reach is None or new_reach < current_reach:
                self._reachability_distance[neighbor_idx] = new_reach
                heapq.heappush(seed_heap, (new_reach, int(neighbor_idx)))

    def _compute_ordering(self, dist_matrix: np.ndarray, bags: List[Bag]):
        """
        Implementa el Paso 1 de COSMIC (Algoritmo 5): calcula el ordenamiento de bolsas
        por densidad, junto con la core-distance y reachability-distance de cada una.
        """
        num_bags = len(bags)
        visited = np.zeros(num_bags, dtype=bool)
        self._core_distance = [None] * num_bags
        self._reachability_distance = [None] * num_bags
        ordering: List[int] = []

        for start in range(num_bags):
            if visited[start]:
                continue

            visited[start] = True
            self._core_distance[start] = self._core_distance_of(start, dist_matrix)
            ordering.append(start)
            logger.debug(f"Procesando bolsa semilla {bags[start].bag_id}")

            if self._core_distance[start] is None:
                continue  # No es núcleo: no se expande vecindad

            seed_heap: list = []
            self._update_seeds(start, self._core_distance[start], dist_matrix, visited, seed_heap)

            while seed_heap:
                _, current = heapq.heappop(seed_heap)
                if visited[current]:
                    continue  # Entrada obsoleta (borrado perezoso)

                visited[current] = True
                self._core_distance[current] = self._core_distance_of(current, dist_matrix)
                ordering.append(current)

                if self._core_distance[current] is not None:
                    self._update_seeds(
                        current, self._core_distance[current], dist_matrix, visited, seed_heap
                    )

        self._ordering = ordering

    def fit(
        self,
        dataset: MIData,
        precomputed_matrix: Optional[np.ndarray] = None,
    ) -> "COSMIC":
        """Ejecuta el Paso 1 de COSMIC (ordenamiento por densidad) y, a continuación,
        extrae automáticamente una partición inicial de clusters (Paso 2) usando
        epsilon_prime (o epsilon, si no se definió epsilon_prime).

        Args:
            dataset: Objeto MIData con las bolsas de entrenamiento.
            precomputed_matrix: Matriz de distancias (N x N) ya calculada. Si se
                proporciona, se omite el cálculo interno. Debe estar alineada con
                ``dataset.bags`` en el mismo orden.

        Returns:
            Instancia del modelo para permitir encadenamiento.

        Raises:
            ValueError: Si el dataset está vacío o la forma de la matriz no coincide
                con el número de bolsas.
        """
        if dataset.get_num_bags() == 0:
            error_msg = "El dataset de entrenamiento está vacío."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if precomputed_matrix is not None:
            n = dataset.get_num_bags()
            if precomputed_matrix.shape != (n, n):
                raise ValueError(
                    f"precomputed_matrix shape {precomputed_matrix.shape} "
                    f"no coincide con n_bags={n}"
                )

        self._reset_state()

        bags = dataset.bags
        self._train_bags = bags

        if precomputed_matrix is not None:
            dist_matrix = precomputed_matrix
            logger.debug("Reutilizando matriz de distancias precomputada.")
        else:
            dist_matrix = self._compute_distance_matrix(bags)

        logger.info(
            f"Calculando ordenamiento COSMIC (eps={self._epsilon}, min_pts={self._min_pts})..."
        )
        self._compute_ordering(dist_matrix, bags)
        self._fitted = True

        # Extracción automática de una partición inicial sobre el ordenamiento recién calculado
        self.extract_clusters(self._epsilon_prime)

        return self

    def extract_clusters(self, epsilon_prime: Optional[float] = None) -> Dict[str, int]:
        """Implementa el Paso 2 de COSMIC (Algoritmo 6): extrae una partición de
        clusters a partir del ordenamiento ya calculado, usando el umbral epsilon_prime.

        A diferencia de DBSCAN, este paso NO requiere recalcular distancias ni el
        ordenamiento: puede llamarse repetidamente con distintos epsilon_prime <= epsilon
        para explorar varias granularidades de clustering sobre un mismo ajuste.

        Args:
            epsilon_prime: Umbral de extracción (epsilon_prime <= epsilon). Si es None,
                se usa el epsilon_prime del constructor, o epsilon en su defecto.

        Returns:
            Diccionario con las etiquetas asignadas {bag_id: cluster_id}.

        Raises:
            RuntimeError: Si no se ha calculado el ordenamiento (no se llamó a fit()).
            ValueError: Si epsilon_prime > epsilon.
        """
        if not self._fitted or self._ordering is None:
            raise RuntimeError(
                "No hay un ordenamiento calculado. Ejecuta .fit() antes de extract_clusters()."
            )

        eps_prime = epsilon_prime if epsilon_prime is not None else self._epsilon_prime
        if eps_prime is None:
            eps_prime = self._epsilon
        if eps_prime > self._epsilon:
            raise ValueError(
                f"epsilon_prime ({eps_prime}) no puede ser mayor que epsilon ({self._epsilon})."
            )

        labels: Dict[str, int] = {}
        current_cluster_id = self.NOISE_LABEL
        next_cluster_id = 0
        self._core_bags = []
        self._core_bag_labels = {}

        for idx in self._ordering:
            bag = self._train_bags[idx]
            reach = self._reachability_distance[idx]
            core = self._core_distance[idx]
            is_core = core is not None and core <= eps_prime

            if reach is None or reach > eps_prime:
                if is_core:
                    current_cluster_id = next_cluster_id
                    next_cluster_id += 1
                else:
                    current_cluster_id = self.NOISE_LABEL
            # si reach <= eps_prime, se hereda el cluster actual (punto denso/borde)

            labels[bag.bag_id] = current_cluster_id

            if is_core and current_cluster_id != self.NOISE_LABEL:
                self._add_core_point(bag, current_cluster_id)

        self._labels = labels
        self._cluster_count = next_cluster_id
        self._epsilon_prime = eps_prime

        num_noise = sum(1 for v in labels.values() if v == self.NOISE_LABEL)
        logger.info(
            f"Extracción completada (epsilon_prime={eps_prime}): "
            f"{next_cluster_id} clusters, {num_noise} bolsas de ruido."
        )

        return self._labels.copy()

    def predict(self, test_dataset: MIData) -> Dict[str, int]:
        """No soportado: COSMIC es un método transductivo.

        A diferencia de MIDBSCAN, el ordenamiento por densidad (Algoritmo 5) depende de
        las relaciones mutuas dentro del propio conjunto de entrenamiento, y el
        algoritmo original (Kriegel et al., 2006) no define un mecanismo para
        posicionar bolsas nuevas dentro de ese ordenamiento sin recalcularlo.

        Raises:
            NotImplementedError: Siempre. Si necesitas evaluar sobre un split
                train/test independiente, considera una extensión heurística propia
                (p. ej. asignar cada bolsa de test al núcleo más cercano almacenado en
                ``_core_bags``, de forma análoga a ``MIDBSCAN.predict()``), o usa
                directamente MIDBSCAN.
        """
        raise NotImplementedError(
            "COSMIC es transductivo: el algoritmo original no define un mecanismo de "
            "predicción para bolsas fuera del conjunto de entrenamiento. Usa MIDBSCAN "
            "si necesitas evaluar sobre un split train/test."
        )

    def fit_predict(self, X: MIData, y: Optional[MIData] = None) -> Dict[str, int]:  # type: ignore
        """
        Entrena el modelo (ordenamiento + extracción inicial) y devuelve las etiquetas.
        Para cumplir con scikit-learn, recibe X y opcionalmente y; sin embargo, dado el
        carácter transductivo de COSMIC, ``y`` no puede usarse como conjunto de test
        independiente y se ignora si se proporciona.
        """
        if y is not None:
            logger.warning("COSMIC es transductivo: el argumento 'y' de fit_predict() se ignora.")
        self.fit(X)
        return self.labels

    def get_cluster_sizes(self) -> Dict[int, int]:
        """Devuelve el conteo de elementos por cluster.

        Returns:
            Diccionario {cluster_id: cantidad}.
        """
        if not self._fitted:
            return {}
        return dict(Counter(self._labels.values()))

    def get_noise_points(self) -> List[str]:
        """Devuelve una lista con los IDs de las bolsas consideradas ruido.

        Returns:
            Lista de strings (bag_ids).
        """
        if not self._fitted:
            return []
        return [bid for bid, label in self._labels.items() if label == self.NOISE_LABEL]

    def get_cluster_members(self, cluster_id: int) -> List[str]:
        """Devuelve los IDs de las bolsas que pertenecen a un cluster específico.

        Args:
            cluster_id: ID del cluster a consultar.

        Returns:
            Lista de bag_ids.
        """
        if not self._fitted:
            return []
        return [bid for bid, label in self._labels.items() if label == cluster_id]

    def get_statistics(self) -> Dict[str, Any]:
        """Genera un reporte completo de estadísticas de la última extracción.

        Returns:
            Diccionario con métricas detalladas.
        """
        if not self._fitted:
            return {"status": "not_fitted"}

        total_points = len(self._labels)
        noise_points = self.get_noise_points()
        num_noise = len(noise_points)
        noise_pct = (num_noise / total_points * 100) if total_points > 0 else 0

        return {
            "epsilon": self._epsilon,
            "min_pts": self._min_pts,
            "epsilon_prime": self._epsilon_prime,
            "total_bags": total_points,
            "num_clusters": self._cluster_count,
            "num_core_points": len(self._core_bags),
            "noise_points_count": num_noise,
            "noise_percentage": noise_pct,
            "cluster_sizes": self.get_cluster_sizes(),
        }

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return (
            f"<COSMIC(epsilon={self._epsilon}, min_pts={self._min_pts}, "
            f"epsilon_prime={self._epsilon_prime}, clusters={self._cluster_count}, "
            f"status={state})>"
        )

    def __str__(self) -> str:
        if not self._fitted:
            return f"COSMIC (Unfitted): eps={self._epsilon}, min_pts={self._min_pts}"

        stats = self.get_statistics()
        return (
            f"COSMIC Model:\n"
            f"  - Config: eps={self._epsilon}, min_pts={self._min_pts}, "
            f"eps_prime={self._epsilon_prime}\n"
            f"  - Status: Fitted on {stats['total_bags']} bags\n"
            f"  - Clusters Found: {self._cluster_count}\n"
            f"  - Core Points: {stats['num_core_points']}\n"
            f"  - Noise: {stats['noise_points_count']} bags ({stats['noise_percentage']:.2f}%)"
        )


##### PRUEBA INDIVIDUAL ########
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # 1. Cargar
        full_data = MIData.from_arff("datasets/musk1.arff")
        train_data, test_data = full_data.split_data(percentage_train=70, seed=42)

        # 2. Instanciar
        cosmic = COSMIC(epsilon=900.0, min_pts=2)  # Musk usa valores altos para Hausdorff

        # 3. Entrenar (ordenamiento + extracción inicial con epsilon_prime = epsilon)
        cosmic.fit(train_data)

        # 4. Ver representación string
        print("\n" + "=" * 50)
        print(cosmic)
        print("=" * 50 + "\n")

        # 5. Explorar otra granularidad SIN recalcular el ordenamiento ni distancias
        cosmic.extract_clusters(epsilon_prime=450.0)
        print("Distribución de clusters (eps'=450.0):", cosmic.get_cluster_sizes())

        # 6. predict() no está soportado: COSMIC es transductivo
        try:
            cosmic.predict(test_data)
        except NotImplementedError as e:
            logger.warning(str(e))

    except Exception as e:
        logger.error(f"Error fatal en la ejecución: {e}")