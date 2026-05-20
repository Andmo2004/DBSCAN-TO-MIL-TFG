from typing import Dict, Callable
from miclustering.distances.hausdorff import hausdorff_distance, hausdorff_distance_min, hausdorff_distance_avg
from miclustering.distances.probability_distribution import cauchy_schwarz_distance, earth_movers_distance, mahalanobis_distance

DISTANCE_REGISTRY: Dict[str, Callable] = {
    'hausdorff': hausdorff_distance,
    'hausdorff_min': hausdorff_distance_min,
    'hausdorff_avg': hausdorff_distance_avg,
    'cauchy_schwarz': cauchy_schwarz_distance,
    'earth_movers': earth_movers_distance,
    'mahalanobis': mahalanobis_distance
}