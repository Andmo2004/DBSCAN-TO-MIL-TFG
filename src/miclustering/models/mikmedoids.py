import logging
from typing import List, Dict, Optional, Callable, Any
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClusterMixin

from miclustering.data.midata import MIData
from miclustering.data.bag import Bag
from miclustering.distances import DISTANCE_REGISTRY
from miclustering.distances.distance_matrix import compute_distance_matrix

logger = logging.getLogger(__name__)

class MIKMedoids(BaseEstimator, ClusterMixin):
    """
    Implementación del algoritmo K-Medoids adaptado para Multi-Instance Learning (MIL),
    utilizando el algoritmo PAM (Partitioning Around Medoids).
    """

    def __init__(self, k: int, metric: str = 'hausdorff', max_iters: int = 100, random_state: Optional[int] = None):
        """Constructor del modelo MIKMedoids.
        
        Args:
            k: Número de clústeres.
            metric: Métrica de distancia a utilizar.
            max_iters: Número máximo de iteraciones.
            random_state: Semilla para la inicialización aleatoria.
        """
        if k < 1:
            raise ValueError(f"El parámetro 'k' debe ser >= 1. Recibido: {k}")
        if max_iters < 1:
            raise ValueError(f"El parámetro 'max_iters' debe ser >= 1. Recibido: {max_iters}")
            
        self._k = k
        self._metric_name = metric.lower()
        self._max_iters = max_iters
        self._random_state = random_state

        self._metric_func = self._get_metric_function(self._metric_name)

        # Estado del modelo
        self._labels: Dict[str, int] = {}
        self._fitted = False
        self._train_bags: List[Bag] = []
        
        # Índices de los medoides en _train_bags
        self._medoid_indices: List[int] = []
        
        # Almacenamos matriz de distancias
        self._distance_matrix: Optional[np.ndarray] = None

        logger.debug(f"MIKMedoids inicializado: k={k}, metric={metric}")

    @property
    def k(self) -> int:
        return self._k

    @property
    def labels(self) -> Dict[str, int]:
        return self._labels.copy()

    @property
    def is_fitted(self) -> bool:
        return self._fitted
        
    @property
    def medoids(self) -> List[Bag]:
        """Devuelve las bolsas que son medoides actuales."""
        if not self._fitted:
            return []
        return [self._train_bags[i] for i in self._medoid_indices]

    def _reset_state(self):
        self._labels = {}
        self._fitted = False
        self._train_bags = []
        self._medoid_indices = []
        self._distance_matrix = None

    def _get_metric_function(self, name: str) -> Callable[[Bag, Bag], float]:
        if name not in DISTANCE_REGISTRY:
            valid_keys = list(DISTANCE_REGISTRY.keys())
            raise ValueError(f"Métrica '{name}' no reconocida. Disponibles: {valid_keys}")
        
        return DISTANCE_REGISTRY[name]

    def _compute_distance_matrix(self, bags: List[Bag]) -> np.ndarray:
        return compute_distance_matrix(bags, self._metric_func, self._metric_name)

    def fit(self, dataset: MIData, precomputed_matrix: Optional[np.ndarray] = None) -> "MIKMedoids":
        """
        Entrena el modelo K-Medoids con el dataset usando el algoritmo PAM.
        
        Args:
            dataset: Objeto MIData con las bolsas de entrenamiento.
            precomputed_matrix: Matriz de distancias (N×N) ya calculada.
                                Si se proporciona, se omite el cálculo interno.
                                Debe estar alineada con dataset.bags en el mismo orden.
        """
        if dataset.get_num_bags() == 0:
            raise ValueError("El dataset de entrenamiento está vacío.")

        if precomputed_matrix is not None:
            n = dataset.get_num_bags()
            if precomputed_matrix.shape != (n, n):
                raise ValueError(
                    f"precomputed_matrix shape {precomputed_matrix.shape} "
                    f"no coincide con n_bags={n}"
                )
            
        if dataset.get_num_bags() < self._k:
            logger.warning(f"El número de bolsas ({dataset.get_num_bags()}) es menor que k ({self._k}). Se ajustará k al número de bolsas.")
            self._k = dataset.get_num_bags()

        self._reset_state()

        self._train_bags = dataset.bags
        num_bags = len(self._train_bags)

        if precomputed_matrix is not None:
            self._distance_matrix = precomputed_matrix
            logger.debug("Reutilizando matriz de distancias precomputada.")
        else:
            self._distance_matrix = self._compute_distance_matrix(self._train_bags)

        # Inicialización de medoides
        rng = np.random.RandomState(self._random_state)
        self._medoid_indices = rng.choice(num_bags, self._k, replace=False).tolist()
        
        cluster_assignments = np.zeros(num_bags, dtype=int)
        
        logger.info(f"Iniciando K-Medoids (k={self._k}, max_iters={self._max_iters})...")

        for iteration in range(self._max_iters):
            # 1. Asignar cada punto al medoide más cercano
            medoid_distances = self._distance_matrix[:, self._medoid_indices]
            new_assignments = np.argmin(medoid_distances, axis=1)
            
            # Comprobar convergencia
            if np.array_equal(cluster_assignments, new_assignments):
                logger.debug(f"Convergencia alcanzada en la iteración {iteration}.")
                break
                
            cluster_assignments = new_assignments
            
            # 2. Actualizar medoides
            new_medoids = []
            for c in range(self._k):
                # Puntos asignados al cluster c
                cluster_points = np.where(cluster_assignments == c)[0]
                
                if len(cluster_points) == 0:
                    # Cluster vacío, mantener medoide anterior
                    new_medoids.append(self._medoid_indices[c])
                    continue
                    
                # Submatriz de distancias para los puntos de este clúster
                cluster_dist_matrix = self._distance_matrix[np.ix_(cluster_points, cluster_points)]
                
                # Encontrar el punto que minimiza la suma de distancias dentro del clúster
                sum_distances = np.sum(cluster_dist_matrix, axis=1)
                best_medoid_idx_in_cluster = np.argmin(sum_distances)
                
                new_medoids.append(cluster_points[best_medoid_idx_in_cluster])
                
            self._medoid_indices = new_medoids
            
        else:
            logger.warning(f"K-Medoids no convergió después de {self._max_iters} iteraciones.")

        self._labels = {self._train_bags[i].bag_id: int(cluster_assignments[i]) for i in range(num_bags)}
        self._fitted = True
        self._distance_matrix = None

        return self

    def predict(self, test_dataset: MIData) -> Dict[str, int]:
        """
        Asigna cada bolsa del test_dataset al medoide más cercano.
        """
        if not self._fitted:
            raise RuntimeError("El modelo debe ser entrenado antes de llamar a predict().")
            
        if test_dataset.get_num_bags() == 0:
            raise ValueError("El dataset de prueba no puede estar vacío.")

        logger.info(f"Prediciendo {test_dataset.get_num_bags()} bolsas de prueba...")

        test_labels = {}
        for test_bag in test_dataset.bags:
            best_dist = float('inf')
            assigned_cluster = -1
            
            for c_idx, medoid_idx in enumerate(self._medoid_indices):
                medoid_bag = self._train_bags[medoid_idx]
                dist = self._metric_func(test_bag, medoid_bag)
                
                if dist < best_dist:
                    best_dist = dist
                    assigned_cluster = c_idx
                    
            test_labels[test_bag.bag_id] = assigned_cluster
            
        return test_labels

    def fit_predict(self, X: MIData, y: Optional[MIData] = None) -> Dict[str, int]:  # type: ignore
        self.fit(X)
        if y is not None:
            return self.predict(y)
        return getattr(self, "labels", {})

    def get_cluster_sizes(self) -> Dict[int, int]:
        if not self._fitted:
            return {}
        return dict(Counter(self._labels.values()))

    def get_statistics(self) -> Dict[str, Any]:
        if not self._fitted:
            return {"status": "not_fitted"}
            
        return {
            "k": self._k,
            "metric": self._metric_name,
            "total_bags": len(self._labels),
            "cluster_sizes": self.get_cluster_sizes(),
            "medoids": [bag.bag_id for bag in self.medoids]
        }

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return f"<MIKMedoids(k={self._k}, metric={self._metric_name}, status={state})>"

    def __str__(self) -> str:
        if not self._fitted:
            return f"MIKMedoids (Unfitted): k={self._k}, metric={self._metric_name}"
            
        stats = self.get_statistics()
        return (f"MIKMedoids Model:\n"
                f"  - Config: k={self._k}, metric={self._metric_name}\n"
                f"  - Status: Fitted on {stats['total_bags']} bags\n"
                f"  - Medoids: {stats['medoids']}\n"
                f"  - Cluster Sizes: {stats['cluster_sizes']}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from miclustering.data.midata import MIData
        full_data = MIData.from_arff("datasets/musk1.arff") 
        train_data, test_data = full_data.split_data(percentage_train=70, seed=42)
        
        model = MIKMedoids(k=2, metric='hausdorff_min')
        model.fit(train_data)
        
        print(model)
        preds = model.predict(test_data)
        print("Predicciones primeras 5:", list(preds.items())[:5])
    except Exception as e:
        logger.error(f"Error: {e}")
