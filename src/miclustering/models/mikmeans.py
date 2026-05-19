import logging
from typing import List, Dict, Optional, Callable, Any
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClusterMixin

from miclustering.data.instance import Instance
from miclustering.data.midata import MIData
from miclustering.data.bag import Bag
from miclustering.distances.hausdorff import hausdorff_distance, hausdorff_distance_min, hausdorff_distance_avg
from miclustering.distances.probability_distribution import cauchy_schwarz_distance, earth_movers_distance, mahalanobis_distance

logger = logging.getLogger(__name__)

class MIKMeans(BaseEstimator, ClusterMixin):
    """
    Implementación del algoritmo KMeans adaptado para Multi-Instance Learning (MIL).
    En este caso, los centroides se representan como Bolsas (Bags) que se construyen
    mediante la agregación de las instancias de las bolsas asignadas a cada clúster.
    """

    def __init__(self, k: int, metric: str = 'hausdorff', max_iters: int = 100, random_state: Optional[int] = None):
        """
        Constructor del modelo MIKMeans.
        
        :param k: (int) Número de clústeres.
        :param metric: (str) Métrica de distancia a utilizar entre bolsas y centroides.
        :param max_iters: (int) Número máximo de iteraciones.
        :param random_state: (int) Semilla para la inicialización aleatoria.
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
        
        # Centroides representados como objetos Bag
        self._centroids: List[Bag] = []

        logger.debug(f"MIKMeans inicializado: k={k}, metric={metric}")

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
    def centroids(self) -> List[Bag]:
        """Devuelve los centroides actuales (que son objetos Bag)."""
        return self._centroids

    def _reset_state(self):
        self._labels = {}
        self._fitted = False
        self._train_bags = []
        self._centroids = []

    def _get_metric_function(self, name: str) -> Callable[[Bag, Bag], float]:
        metrics_registry = {
            'hausdorff': hausdorff_distance,
            'hausdorff_min': hausdorff_distance_min,
            'hausdorff_avg': hausdorff_distance_avg,
            'cauchy_schwarz': cauchy_schwarz_distance,
            'earth_movers': earth_movers_distance,
            'mahalanobis': mahalanobis_distance
        }

        if name not in metrics_registry:
            valid_keys = list(metrics_registry.keys())
            raise ValueError(f"Métrica '{name}' no reconocida. Disponibles: {valid_keys}")
        
        return metrics_registry[name]

    def _calculate_centroid(self, cluster_bags: List[Bag], cluster_id: int) -> Bag:
        """
        Calcula el centroide de un conjunto de bolsas mediante la agregación de sus instancias.
        Calcula la media de todas las instancias dentro de las bolsas para crear un vector representativo,
        y lo encapsula en un objeto Bag de una sola instancia, para permitir el cálculo
        de distancias usando las métricas estándar de MIL.
        """
        if not cluster_bags:
            raise ValueError(f"No se puede calcular el centroide del clúster {cluster_id} vacío.")
            
        # Agregamos todas las instancias de todas las bolsas del clúster
        all_instances = np.vstack([bag.as_matrix() for bag in cluster_bags])
        
        # El centroide es la media de todas las instancias
        centroid_vector = np.mean(all_instances, axis=0)
        
        # Envolvemos en Instance para respetar el contrato de Bag
        centroid_instance = Instance(
            values=centroid_vector.tolist(),
            schema=cluster_bags[0].instances[0]._schema
        )        
        
        return Bag(
            bag_id=f"centroid_{cluster_id}",
            label=None,
            instances=[centroid_instance]
        )

    def fit(self, dataset: MIData) -> "MIKMeans":
        """
        Entrena el modelo KMeans adaptado a MIL.
        """
        if dataset.get_num_bags() == 0:
            raise ValueError("El dataset de entrenamiento está vacío.")
            
        if dataset.get_num_bags() < self._k:
            logger.warning(f"El número de bolsas ({dataset.get_num_bags()}) es menor que k ({self._k}). Se ajustará k.")
            self._k = dataset.get_num_bags()

        self._reset_state()
        self._train_bags = dataset.bags
        num_bags = len(self._train_bags)

        # Inicialización de centroides (elegimos k bolsas aleatorias inicialmente)
        rng = np.random.RandomState(self._random_state)
        initial_indices = rng.choice(num_bags, self._k, replace=False).tolist()
        
        # Los centroides iniciales serán la media de las instancias de cada bolsa inicial
        self._centroids = []
        for c_idx, idx in enumerate(initial_indices):
            bag = self._train_bags[idx]
            # Convertimos la bolsa seleccionada a un centroide válido (una bolsa con la media de sus instancias)
            centroid_bag = self._calculate_centroid([bag], c_idx)
            self._centroids.append(centroid_bag)
        
        cluster_assignments = np.zeros(num_bags, dtype=int)
        
        logger.info(f"Iniciando MIKMeans (k={self._k}, max_iters={self._max_iters})...")

        for iteration in range(self._max_iters):
            new_assignments = np.zeros(num_bags, dtype=int)
            
            # 1. Asignar cada bolsa al centroide más cercano
            for i, bag in enumerate(self._train_bags):
                distances = [self._metric_func(bag, centroid) for centroid in self._centroids]
                new_assignments[i] = np.argmin(distances)
                
            # Comprobar convergencia
            if np.array_equal(cluster_assignments, new_assignments):
                logger.debug(f"Convergencia alcanzada en la iteración {iteration}.")
                break
                
            cluster_assignments = new_assignments
            
            # 2. Actualizar centroides
            new_centroids = []
            for c in range(self._k):
                cluster_points = np.where(cluster_assignments == c)[0]
                
                if len(cluster_points) == 0:
                    # Clúster vacío: mantener el centroide anterior o reasignar aleatoriamente
                    logger.debug(f"Clúster {c} vacío en iteración {iteration}.")
                    new_centroids.append(self._centroids[c])
                    continue
                    
                cluster_bags = [self._train_bags[idx] for idx in cluster_points]
                new_centroids.append(self._calculate_centroid(cluster_bags, c))
                
            self._centroids = new_centroids
            
        else:
            logger.warning(f"MIKMeans no convergió después de {self._max_iters} iteraciones.")

        self._labels = {self._train_bags[i].bag_id: int(cluster_assignments[i]) for i in range(num_bags)}
        self._fitted = True
        
        return self

    def predict(self, test_dataset: MIData) -> Dict[str, int]:
        """
        Asigna cada bolsa del test_dataset al centroide más cercano.
        """
        if not self._fitted:
            raise RuntimeError("El modelo debe ser entrenado antes de llamar a predict().")
            
        if test_dataset.get_num_bags() == 0:
            raise ValueError("El dataset de prueba no puede estar vacío.")

        logger.info(f"Prediciendo {test_dataset.get_num_bags()} bolsas de prueba...")

        test_labels = {}
        for test_bag in test_dataset.bags:
            distances = [self._metric_func(test_bag, centroid) for centroid in self._centroids]
            test_labels[test_bag.bag_id] = np.argmin(distances)
            
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
            "cluster_sizes": self.get_cluster_sizes()
        }

    def __repr__(self,  N_CHAR_MAX: int = 700) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return f"<MIKMeans(k={self._k}, metric={self._metric_name}, status={state})>"

    def __str__(self) -> str:
        if not self._fitted:
            return f"MIKMeans (Unfitted): k={self._k}, metric={self._metric_name}"
            
        stats = self.get_statistics()
        return (f"MIKMeans Model:\n"
                f"  - Config: k={self._k}, metric={self._metric_name}\n"
                f"  - Status: Fitted on {stats['total_bags']} bags\n"
                f"  - Cluster Sizes: {stats['cluster_sizes']}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from miclustering.data.midata import MIData
        full_data = MIData.from_arff("datasets/musk1.arff") 
        train_data, test_data = full_data.split_data(percentage_train=70, seed=42)
        
        model = MIKMeans(k=2, metric='hausdorff_avg')
        model.fit(train_data)
        
        print(model)
        preds = model.predict(test_data)
        print("Predicciones primeras 5:", list(preds.items())[:5])
    except Exception as e:
        logger.error(f"Error: {e}")
