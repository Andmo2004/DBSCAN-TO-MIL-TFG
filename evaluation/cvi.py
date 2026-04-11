"""
evaluation/internal_cvi.py
 
Métricas de Validación Interna de Clustering (CVIs internos) para MIL.
 
Los CVIs internos evalúan la calidad del clustering basándose únicamente en
la estructura de los datos y la agrupación obtenida, SIN usar etiquetas reales.
Son especialmente útiles para comparar configuraciones de parámetros cuando
no se dispone de ground-truth.
 
Todas las métricas trabajan con la MATRIZ DE DISTANCIAS precomputada entre
bolsas, lo que las hace agnósticas al espacio de features original — requisito
fundamental en MIL donde no existe un espacio vectorial canónico de bolsas.

Tipo 1 — Solo Compactibilidad (↓ mejor):
    - SED  : Suma de Distancias Euclidianas
    - DD   : Distancia Distorsionada (SED normalizada por n y d)
    - Hc   : Entropía de distribución de tamaños de clusters

Tipo 2 — Compactibilidad + Separación:
Tipo 3 — Estructuras Especiales / orientado a densidad:

"""

import logging
import numpy as np

from data.midata import MIData
from typing import Any, Dict, List, Optional, Tuple

_NOISE = -1

logger = logging.getLogger(__name__)

class BaseCVI:
    """
    Clase base para los Índices de Validación Interna de Clustering.
    """
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        """Nombre legible del índice."""
        raise NotImplementedError("Las subclases deben implementar name")

    @property
    def category(self) -> str:
        """
        Grupo de clasificación:
          'compactness'            → solo compactibilidad
          'compactness_separation' → compactibilidad + separación
          'special'                → estructuras especiales / densidad
        """
        raise NotImplementedError("Las subclases deben implementar category")

    @property
    def higher_is_better(self) -> bool:
        """
        True si valores mayores indican mejor clustering.
        Por defecto True; las métricas que minimizan deben sobreescribirlo.
        """
        return True  # valor por defecto, las subclases pueden sobreescribirlo

    def compute(        
        self,
        dist_matrix: np.ndarray,
        labels: Dict[str, int],
        bag_ids: List[str],
        X: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calcula el índice de validación interna.
 
        :param dist_matrix: Matriz cuadrada (N×N) de distancias entre bolsas.
                            Debe ser simétrica y con diagonal 0.
        :param labels:      {bag_id: cluster_id}. Ruido → -1.
        :param bag_ids:     Lista de bag_ids en el mismo orden que dist_matrix.
        :returns:           Valor escalar del índice (float).
        """
        raise NotImplementedError("Las subclases deben implementar compute()")
    
    # ── Utilidades internas compartidas ──────────────────────────────────────
    
    def _label_array(self, labels: Dict[str, int], bag_ids: List[str]) -> np.ndarray:
        """Convierte el dict de etiquetas a array numpy alineado con bag_ids."""
        return np.array([labels.get(bid, _NOISE) for bid in bag_ids])
 
    def _real_clusters(self, label_arr: np.ndarray) -> np.ndarray:
        """IDs de clusters reales (excluye ruido -1)."""
        return np.unique(label_arr[label_arr >= 0])
 
    def _cluster_idx(self, label_arr: np.ndarray, cid: int) -> np.ndarray:
        """Índices (posición en dist_matrix / X) de los miembros de un cluster."""
        return np.where(label_arr == cid)[0]
 
    def _require_X(self, X: Optional[np.ndarray]) -> np.ndarray:
        """Lanza ValueError claro si X no fue proporcionada."""
        if X is None:
            raise ValueError(
                f"{self.name} necesita la matriz de características X "
                "(centroides de bolsas). Pasa dataset a InternalCVIEvaluator.evaluate()."
            )
        return X
 
    def __repr__(self) -> str:
        arrow = "↑" if self.higher_is_better else "↓"
        return f"<{self.__class__.__name__} [{self.category}] {arrow} mejor>"
    

# ══════════════════════════════════════════════════════════════════════════════
# TIPO 1 — Solo Compactibilidad
# ══════════════════════════════════════════════════════════════════════════════

class SEDIndex(BaseCVI):
    """
    Suma de Distancias Euclidianas (SED) — ↓ mejor.
 
    Mide la dispersión total de las instancias respecto al centroide de su
    cluster. Un SED bajo indica clusters compactos y bien cohesionados.
 
    Fórmula (Tabla 4.2, ec. 1):
        SED = Σ_Cj  Σ_{xi ∈ Cj}  ||xi - µj||
 
    Donde:
      xi  = centroide de la bolsa i (fila i de X).
      µj  = centroide del cluster Cj = media de los xi que pertenecen a Cj.
 
    Notas:
      - Los puntos de ruido (-1) se excluyen del cálculo.
      - Clusters singleton contribuyen 0 (||xi - xi|| = 0).
      - Es una función monótona decreciente con k: a más clusters, menor SED.
        Por eso es más útil para comparar configuraciones con el MISMO k.
      - Requiere X (centroides de bolsas). InternalCVIEvaluator lo calcula
        automáticamente si se pasa dataset.  
    """

    @property
    def name(self) -> str:
        return "SED"
 
    @property
    def category(self) -> str:
        return "compactness"
 
    @property
    def higher_is_better(self) -> bool:
        return False   
    
    def compute(
        self,
        dist_matrix: np.ndarray,
        labels: Dict[str, int],
        bag_ids: List[str],
        X: Optional[np.ndarray] = None,
    ) -> float:
        X         = self._require_X(X)
        label_arr = self._label_array(labels, bag_ids)
        clusters  = self._real_clusters(label_arr)
 
        if len(clusters) == 0:
            logger.warning("[SED] No hay clusters reales.")
            return float("inf")
 
        sed = 0.0
        for cid in clusters:
            idx = self._cluster_idx(label_arr, int(cid))
            if len(idx) == 0:
                continue
            mu_j  = X[idx].mean(axis=0)            # (n_features,)
            diffs = X[idx] - mu_j                  # (|Cj|, n_features)
            sed  += float(np.linalg.norm(diffs, axis=1).sum())
 
        return sed
    

class DDIndex(BaseCVI):
    """
    Distancia Distorsionada (DD) — ↓ mejor.
 
    Versión normalizada de SSE (suma del error cuadrático) que divide por
    el número total de instancias n y la dimensionalidad d, haciéndola
    comparable entre datasets de distinto tamaño y dimensión.
 
    Fórmula (Tabla 4.2, ec. 5):
        DD = Σ_Cj  Σ_{xi ∈ Cj}  ||xi - µj||²  /  (n · d)
 
    Donde:
      xi  = centroide de la bolsa i.
      µj  = centroide del cluster Cj.
      n   = número total de bolsas válidas (excluye ruido).
      d   = dimensionalidad de las bolsas (número de features).
 
    Notas:
      - Al normalizar por n·d, DD es directamente comparable entre
        experimentos con distintos datasets o dimensionalidades.
      - Como SED, es monótona decreciente con k.
      - Requiere X.
 
    Ref: Gomez-Flores (tesis), ec. (5).
    """
 
    @property
    def name(self) -> str:
        return "DD"
 
    @property
    def category(self) -> str:
        return "compactness"
 
    @property
    def higher_is_better(self) -> bool:
        return False
 
    def compute(
        self,
        dist_matrix: np.ndarray,
        labels: Dict[str, int],
        bag_ids: List[str],
        X: Optional[np.ndarray] = None,
    ) -> float:
        X         = self._require_X(X)
        label_arr = self._label_array(labels, bag_ids)
        clusters  = self._real_clusters(label_arr)
 
        if len(clusters) == 0:
            logger.warning("[DD] No hay clusters reales.")
            return float("inf")
 
        n_valid = int(np.sum(label_arr >= 0))
        d       = X.shape[1]
 
        if n_valid == 0 or d == 0:
            return float("inf")
 
        sse = 0.0
        for cid in clusters:
            idx  = self._cluster_idx(label_arr, int(cid))
            if len(idx) == 0:
                continue
            mu_j  = X[idx].mean(axis=0)
            diffs = X[idx] - mu_j
            sse  += float((np.linalg.norm(diffs, axis=1) ** 2).sum())
 
        return sse / (n_valid * d)
 
 
class HcIndex(BaseCVI):
    """
    Entropía de distribución de tamaños de clusters (Hc) — ↓ mejor.
 
    La entropía de Bezdek (1981) está definida originalmente para clustering
    difuso. En este proyecto se adapta a clustering duro (DBSCAN) como la
    entropía de Shannon de la distribución de probabilidad de los tamaños
    de cluster:
 
        p_k  = |Ck| / n_valid          (proporción del cluster k)
        Hc   = -Σ_k  p_k · log(p_k)   (en nats)
 
    Interpretación:
      - Hc = 0     → un único cluster (o todos del mismo tamaño unitario).
      - Hc máximo  → todos los clusters tienen exactamente el mismo tamaño
                     (distribución uniforme, máximo desorden).
      - Valores bajos indican que pocos clusters concentran la mayoría de los
        puntos, lo que en DBSCAN suele corresponder a una partición más clara.
      - Puntos de ruido (-1) se excluyen: no pertenecen a ningún cluster.
 
    No requiere X.
 
    Ref: Bezdek (1981); Gomez-Flores (tesis), ec. (8).
    """
 
    @property
    def name(self) -> str:
        return "Hc"
 
    @property
    def category(self) -> str:
        return "compactness"
 
    @property
    def higher_is_better(self) -> bool:
        return False
 
    def compute(
        self,
        dist_matrix: np.ndarray,
        labels: Dict[str, int],
        bag_ids: List[str],
        X: Optional[np.ndarray] = None,    # no se usa, firma uniforme
    ) -> float:
        label_arr = self._label_array(labels, bag_ids)
        clusters  = self._real_clusters(label_arr)
 
        if len(clusters) == 0:
            logger.warning("[Hc] No hay clusters reales.")
            return float("inf")
 
        n_valid = int(np.sum(label_arr >= 0))
        if n_valid == 0:
            return float("inf")
 
        hc = 0.0
        for cid in clusters:
            nk = len(self._cluster_idx(label_arr, int(cid)))
            if nk == 0:
                continue
            p_k  = nk / n_valid
            hc  -= p_k * np.log(p_k)     # entropía de Shannon en nats
 
        return float(hc)
    
# ══════════════════════════════════════════════════════════════════════════════
# TIPO 2 — Compactibilidad + Separación
# ══════════════════════════════════════════════════════════════════════════════

class VRCIndex(BaseCVI):
    """
    Criterio de Relación de Varianza (VRC) o Índice Calinski-Harabasz — ↑ mejor.
 
    Mide la cohesión interna de los clusters y su aislamiento respecto al
    resto, comparando la varianza entre clusters (SSB) con la varianza dentro
    de los clusters (SSW):
 
        VRC = (SSB / SSW) · (n - k) / (k - 1)
 
        SSB = Σ_k |Cj| · ||µj - M||²        (varianza entre clusters)
        SSW = Σ_k Σ_{xi∈Cj} ||xi - µj||²    (varianza dentro de clusters)
        M   = centroide global de todos los puntos válidos (sin ruido)
        n   = número de puntos válidos
        k   = número de clusters reales
 
    Propiedades:
      - Valores altos → clusters compactos y bien separados entre sí.
      - Indefinido para k=1 (división por k-1=0); se devuelve 0.0.
      - Los puntos de ruido (-1) se excluyen de SSB, SSW y del centroide M.
      - Es la base del índice CH (Calinski-Harabasz) ampliamente usado en
        scikit-learn como `calinski_harabasz_score`.
 
    Requiere X.
 
    Ref: Caliński & Harabasz (1974); Gomez-Flores (tesis), ec. (19).
    """
 
    @property
    def name(self) -> str:
        return "VRC"
 
    @property
    def category(self) -> str:
        return "compactness_separation"
 
    def compute(
        self,
        dist_matrix: np.ndarray,
        labels: Dict[str, int],
        bag_ids: List[str],
        X: Optional[np.ndarray] = None,
    ) -> float:
        X         = self._require_X(X)
        label_arr = self._label_array(labels, bag_ids)
        clusters  = self._real_clusters(label_arr)
        k         = len(clusters)
 
        if k < 2:
            logger.warning("[VRC] Se necesitan al menos 2 clusters.")
            return 0.0
 
        valid_mask = label_arr >= 0
        n_valid    = int(valid_mask.sum())
 
        if n_valid == 0:
            return 0.0
 
        # Centroide global M (solo puntos válidos)
        M = X[valid_mask].mean(axis=0)   # (n_features,)
 
        # SSB: varianza entre clusters
        ssb = 0.0
        for cid in clusters:
            idx  = self._cluster_idx(label_arr, int(cid))
            mu_j = X[idx].mean(axis=0)
            ssb += len(idx) * float(np.dot(mu_j - M, mu_j - M))
 
        # SSW: varianza dentro de clusters
        ssw = 0.0
        for cid in clusters:
            idx   = self._cluster_idx(label_arr, int(cid))
            mu_j  = X[idx].mean(axis=0)
            diffs = X[idx] - mu_j
            ssw  += float((diffs ** 2).sum())
 
        if ssw < 1e-12:
            # Clusters perfectamente compactos (todos singletons idénticos)
            return float("inf")
 
        return float((ssb / ssw) * ((n_valid - k) / (k - 1)))
 
 
class IIndex(BaseCVI):
    """
    Índice I (o PBM — Pakhira-Bandyopadhyay-Maulik) — ↑ mejor.
 
    Combina tres factores para penalizar a la vez la fragmentación (muchos
    clusters), la dispersión interna y la falta de separación entre clusters:
 
        I(k) = (1/k · E1/Ek · Dk)^p      p = 2
 
        E1 = Σ_i ||xi - M||              (SED respecto al centroide global)
        Ek = Σ_k Σ_{xi∈Cj} ||xi - µj||  (SED respecto a centroides de cluster)
        Dk = max_{j≠j'} ||µj - µj'||     (máxima separación entre centroides)
        M  = centroide global de todos los puntos válidos
 
    Interpretación de los tres factores:
      - 1/k          : penaliza tener muchos clusters.
      - E1/Ek        : penaliza que los clusters sean dispersos (Ek alto)
                       y premia que el dataset esté concentrado (E1 bajo).
                       Si Ek → 0 los clusters son perfectamente compactos.
      - Dk           : premia la separación máxima entre centroides.
 
    Propiedades:
      - Un único cluster (k=1) devuelve 0.0 (Dk = 0, no hay separación).
      - Si Ek ≈ 0 (clusters degenerados de un solo punto), devuelve inf.
      - Los puntos de ruido se excluyen.
 
    Requiere X.
 
    Ref: Pakhira, Bandyopadhyay & Maulik (2004); Gomez-Flores (tesis), ec. (21).
    """
 
    _P = 2  # exponente fijo según los autores
 
    @property
    def name(self) -> str:
        return "I"
 
    @property
    def category(self) -> str:
        return "compactness_separation"
 
    def compute(
        self,
        dist_matrix: np.ndarray,
        labels: Dict[str, int],
        bag_ids: List[str],
        X: Optional[np.ndarray] = None,
    ) -> float:
        X         = self._require_X(X)
        label_arr = self._label_array(labels, bag_ids)
        clusters  = self._real_clusters(label_arr)
        k         = len(clusters)
 
        if k == 0:
            logger.warning("[IIndex] No hay clusters reales.")
            return 0.0
 
        valid_mask = label_arr >= 0
 
        if not valid_mask.any():
            return 0.0
 
        # Centroide global M
        M = X[valid_mask].mean(axis=0)   # (n_features,)
 
        # E1: SED de todos los puntos válidos respecto a M
        diffs_global = X[valid_mask] - M
        e1 = float(np.linalg.norm(diffs_global, axis=1).sum())
 
        # Ek: SED de cada punto respecto al centroide de su cluster
        ek = 0.0
        centroids: List[np.ndarray] = []
        for cid in clusters:
            idx  = self._cluster_idx(label_arr, int(cid))
            mu_j = X[idx].mean(axis=0)
            centroids.append(mu_j)
            diffs = X[idx] - mu_j
            ek   += float(np.linalg.norm(diffs, axis=1).sum())
 
        if ek < 1e-12:
            logger.warning("[IIndex] Ek ≈ 0: clusters degenerados.")
            return float("inf")
 
        # Dk: máxima distancia euclídea entre pares de centroides
        dk = 0.0
        for a in range(len(centroids)):
            for b in range(a + 1, len(centroids)):
                d = float(np.linalg.norm(centroids[a] - centroids[b]))
                if d > dk:
                    dk = d
 
        if dk < 1e-12:
            # Todos los centroides son idénticos → no hay separación
            return 0.0
 
        return float(((1.0 / k) * (e1 / ek) * dk) ** self._P)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluador Unificado de CVIs Internos
# ══════════════════════════════════════════════════════════════════════════════
 
class InternalCVIEvaluator:
    """
    Orquestador que ejecuta y reporta todos los CVIs internos registrados.
 
    Uso básico (calcula X automáticamente a partir del dataset):
        evaluator = InternalCVIEvaluator()
        results   = evaluator.evaluate(dist_matrix, labels, bag_ids,
                                       dataset=train_scaled)
 
    Uso sin dataset (solo CVIs que no necesitan X, como Hc):
        results = evaluator.evaluate(dist_matrix, labels, bag_ids)
 
    Uso personalizado (solo algunos CVIs):
        evaluator = InternalCVIEvaluator(cvis=[SEDIndex(), HcIndex()])
 
    """
 
    _DEFAULT_CVIS: List[BaseCVI] = [
        SEDIndex(),
        DDIndex(),
        HcIndex(),
    ]
 
    def __init__(self, cvis: Optional[List[BaseCVI]] = None) -> None:
        """
        :param cvis: Lista de instancias BaseCVI a ejecutar.
                     None → usa todos los CVIs registrados por defecto.
        """
        self._cvis: List[BaseCVI] = (
            list(cvis) if cvis is not None else list(self._DEFAULT_CVIS)
        )
 
    # ── API pública ───────────────────────────────────────────────────────────
 
    def register(self, cvi: BaseCVI) -> "InternalCVIEvaluator":
        """Añade un CVI al evaluador. Retorna self para encadenamiento."""
        if not isinstance(cvi, BaseCVI):
            raise TypeError(
                f"Se esperaba una instancia de BaseCVI, recibido: {type(cvi)}"
            )
        self._cvis.append(cvi)
        return self
 
    @property
    def cvi_names(self) -> List[str]:
        """Nombres de los CVIs registrados."""
        return [cvi.name for cvi in self._cvis]
 
    def evaluate(
        self,
        dist_matrix: np.ndarray,
        labels: Dict[str, int],
        bag_ids: List[str],
        dataset: Optional[MIData] = None,
        title: str = "Evaluación Interna",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Calcula todos los CVIs registrados y devuelve un diccionario de resultados.
 
        :param dist_matrix: Matriz (N×N) de distancias entre bolsas.
        :param labels:      {bag_id: cluster_id}; ruido → -1.
        :param bag_ids:     Lista de bag_ids en el orden de dist_matrix.
        :param dataset:     MIData opcional. Si se proporciona, se calcula X
                            automáticamente como el centroide (mean) de cada bolsa.
                            Necesario para SED y DD.
        :param title:       Título para el reporte en consola.
        :param verbose:     Si True, imprime reporte formateado.
        :returns:
            Dict con claves:
              "title"      : str
              "n_bags"     : int
              "n_clusters" : int
              "noise_count": int
              "noise_pct"  : float
              "scores"     : Dict[name → {"value", "category",
                                          "higher_is_better", ?"error"}]
        """
        # ── Calcular X si se proporcionó dataset ──────────────────────────────
        X: Optional[np.ndarray] = None
        if dataset is not None:
            X = self._compute_bag_centroids(dataset, bag_ids)
 
        # ── Estadísticas generales ────────────────────────────────────────────
        label_arr     = np.array([labels.get(bid, _NOISE) for bid in bag_ids])
        real_clusters = np.unique(label_arr[label_arr >= 0])
        noise_count   = int(np.sum(label_arr < 0))
        n             = len(bag_ids)
 
        results: Dict[str, Any] = {
            "title":       title,
            "n_bags":      n,
            "n_clusters":  len(real_clusters),
            "noise_count": noise_count,
            "noise_pct":   round(100.0 * noise_count / n, 2) if n > 0 else 0.0,
            "scores":      {},
        }
 
        # ── Ejecutar cada CVI ─────────────────────────────────────────────────
        for cvi in self._cvis:
            try:
                value = cvi.compute(dist_matrix, labels, bag_ids, X=X)
                results["scores"][cvi.name] = {
                    "value":            round(float(value), 6),
                    "category":         cvi.category,
                    "higher_is_better": cvi.higher_is_better,
                }
                logger.debug(f"[InternalCVI] {cvi.name}: {value:.6f}")
 
            except Exception as exc:
                logger.warning(f"[InternalCVI] Error en {cvi.name}: {exc}")
                results["scores"][cvi.name] = {
                    "value":            None,
                    "category":         cvi.category,
                    "higher_is_better": cvi.higher_is_better,
                    "error":            str(exc),
                }
 
        if verbose:
            self._print_report(results)
 
        return results
 
    # ── Cálculo automático de centroides de bolsas ────────────────────────────
 
    @staticmethod
    def _compute_bag_centroids(dataset: MIData, bag_ids: List[str]) -> np.ndarray:
        """
        Calcula el centroide de cada bolsa como la media de sus instancias.
 
        El orden de filas en X respeta el orden de bag_ids, que es el mismo
        que el de dist_matrix, garantizando la alineación de índices.
 
        :param dataset:  MIData (ya escalado).
        :param bag_ids:  Lista de bag_ids en el orden de dist_matrix.
        :returns:        np.ndarray (N × n_features).
        """
        bag_index = {bag.bag_id: bag for bag in dataset.bags}
 
        centroids = []
        for bid in bag_ids:
            bag = bag_index.get(bid)
            if bag is None or len(bag) == 0:
                d = len(centroids[0]) if centroids else 1
                centroids.append(np.zeros(d))
                logger.warning(
                    f"[InternalCVIEvaluator] Bolsa '{bid}' no encontrada o vacía."
                )
            else:
                centroids.append(np.mean(bag.as_matrix(), axis=0))
 
        return np.array(centroids)   # (N, n_features)
 
    # ── Reporte en consola ────────────────────────────────────────────────────
 
    @staticmethod
    def _print_report(results: Dict[str, Any]) -> None:
        W = 65
 
        scores    = results["scores"]
        title     = results["title"]
        n         = results["n_bags"]
        nc        = results["n_clusters"]
        noise     = results["noise_count"]
        noise_pct = results["noise_pct"]
 
        print(f"\n{'═'*W}")
        print(f"  CVIs INTERNOS — {title}")
        print(f"{'═'*W}")
        print(f"  Bolsas totales : {n}")
        print(f"  Clusters reales: {nc}")
        print(f"  Ruido          : {noise} bolsas ({noise_pct:.1f}%)")
 
        groups = [
            ("compactness",            "GRUPO 1 — Solo Compactibilidad"),
            ("compactness_separation", "GRUPO 2 — Compactibilidad + Separación"),
            ("special",                "GRUPO 3 — Estructuras Especiales"),
        ]
 
        for cat_key, cat_label in groups:
            cat_items = {
                name: info for name, info in scores.items()
                if info.get("category") == cat_key
            }
            if not cat_items:
                continue
 
            print(f"\n  {'─'*W}")
            print(f"  {cat_label}")
            print(f"  {'─'*W}")
            print(f"  {'Índice':<20} {'Valor':>12}   {'Criterio'}")
            print(f"  {'·'*50}")
 
            for name, info in cat_items.items():
                val = info.get("value")
                if val is None:
                    val_str = f"  ERROR: {info.get('error', '?')}"
                elif abs(val) >= 1e15:
                    val_str = f"{'∞':>12}"
                else:
                    val_str = f"{val:>12.6f}"
 
                arrow = "↑ mayor mejor" if info["higher_is_better"] else "↓ menor mejor"
                print(f"  {name:<20} {val_str}   {arrow}")
 
        print(f"\n{'═'*W}\n")