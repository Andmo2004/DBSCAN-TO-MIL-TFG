"""
miclustering.distances.torch_backend

Módulo de aceleración por hardware (GPU CUDA, Apple Silicon MPS y CPU Vectorial)
para distancias Multi-Instance Learning mediante PyTorch.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Any, Callable
import numpy as np

from miclustering.data.bag import Bag

logger = logging.getLogger(__name__)

# Intento de importar PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

# Intento de importar POT (Python Optimal Transport)
try:
    import ot
    POT_AVAILABLE = True
except ImportError:
    ot = None  # type: ignore
    POT_AVAILABLE = False


def is_torch_available() -> bool:
    """Verifica si PyTorch está instalado y disponible en el entorno."""
    return TORCH_AVAILABLE


def get_torch_device(device_str: str = "auto") -> Optional["torch.device"]:
    """Resuelve y devuelve el objeto torch.device apropiado según la disponibilidad.

    Args:
        device_str: 'auto', 'cuda', 'mps', 'cpu' o None.

    Returns:
        torch.device o None si PyTorch no está disponible.
    """
    if not TORCH_AVAILABLE:
        return None

    dev = (device_str or "auto").lower().strip()

    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    try:
        return torch.device(dev)
    except Exception as e:
        logger.warning(f"Dispositivo solicitado '{device_str}' no válido ({e}). Usando CPU.")
        return torch.device("cpu")


#  Distancias Aceleradas por GPU 

def hausdorff_torch(
    mat1: np.ndarray,
    mat2: np.ndarray,
    device: Optional[Any] = None,
    mode: str = "max",
) -> float:
    """Calcula la distancia de Hausdorff (Max, Min o Avg) usando tensores PyTorch en GPU.

    Args:
        mat1: Matriz de la primera bolsa (n_inst x d).
        mat2: Matriz de la segunda bolsa (m_inst x d).
        device: torch.device para el cálculo.
        mode: 'max', 'min' o 'avg'.

    Returns:
        Distancia de Hausdorff (float).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch no está disponible en este entorno.")

    if len(mat1) == 0 or len(mat2) == 0:
        return float("inf")

    dev = device or get_torch_device("auto")
    t1 = torch.as_tensor(mat1, dtype=torch.float32, device=dev)
    t2 = torch.as_tensor(mat2, dtype=torch.float32, device=dev)

    # torch.cdist utiliza GEMM optimizado en GPU
    d_mat = torch.cdist(t1, t2, p=2.0)

    if mode == "max":
        h_a_b = torch.max(torch.min(d_mat, dim=1).values)
        h_b_a = torch.max(torch.min(d_mat, dim=0).values)
        return float(torch.maximum(h_a_b, h_b_a).item())
    elif mode == "min":
        return float(torch.min(d_mat).item())
    elif mode == "avg":
        sum_a_b = torch.sum(torch.min(d_mat, dim=1).values)
        sum_b_a = torch.sum(torch.min(d_mat, dim=0).values)
        total = t1.shape[0] + t2.shape[0]
        return float(((sum_a_b + sum_b_a) / total).item())
    else:
        raise ValueError(f"Modo Hausdorff no reconocido: '{mode}'")


def cauchy_schwarz_torch(
    mat1: np.ndarray,
    mat2: np.ndarray,
    device: Optional[Any] = None,
) -> float:
    """Calcula la distancia Cauchy-Schwarz vectorizada en GPU con PyTorch."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch no está disponible en este entorno.")

    if len(mat1) == 0 or len(mat2) == 0:
        return float("inf")

    dev = device or get_torch_device("auto")
    t1 = torch.as_tensor(mat1, dtype=torch.float32, device=dev)
    t2 = torch.as_tensor(mat2, dtype=torch.float32, device=dev)

    vec1 = torch.mean(t1, dim=0)
    vec2 = torch.mean(t2, dim=0)

    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return float("inf")

    cos_sim = torch.clamp(torch.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
    return float((1.0 - cos_sim).item())


def mahalanobis_torch(
    mat1: np.ndarray,
    mat2: np.ndarray,
    device: Optional[Any] = None,
) -> float:
    """Calcula la distancia de Mahalanobis vectorizada en GPU con PyTorch."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch no está disponible en este entorno.")

    if len(mat1) == 0 or len(mat2) == 0:
        return float("inf")

    dev = device or get_torch_device("auto")
    dtype = torch.float32 if getattr(dev, "type", "") == "mps" else torch.float64
    t1 = torch.as_tensor(mat1, dtype=dtype, device=dev)
    t2 = torch.as_tensor(mat2, dtype=dtype, device=dev)
    d = t1.shape[1]

    mu_a = torch.mean(t1, dim=0)
    mu_b = torch.mean(t2, dim=0)
    diff = mu_a - mu_b

    # Covarianzas
    cov_a = torch.cov(t1.T) if t1.shape[0] >= 2 else torch.eye(d, dtype=dtype, device=dev)
    cov_b = torch.cov(t2.T) if t2.shape[0] >= 2 else torch.eye(d, dtype=dtype, device=dev)

    cov_comb = 0.5 * cov_a + 0.5 * cov_b + torch.eye(d, dtype=dtype, device=dev) * 1e-5

    try:
        cov_inv = torch.linalg.pinv(cov_comb)
    except Exception:
        cov_inv = torch.eye(d, dtype=dtype, device=dev)

    maha_sq = diff @ cov_inv @ diff

    if not torch.isfinite(maha_sq):
        return float(torch.norm(diff).item())

    return float(torch.sqrt(torch.clamp(maha_sq, min=0.0)).item())


def sinkhorn_emd_torch(
    mat1: np.ndarray,
    mat2: np.ndarray,
    device: Optional[Any] = None,
    reg: Optional[float] = None,
    max_iter: int = 100,
) -> float:
    """Calcula la distancia de Transporte Óptimo (EMD/Sinkhorn) acelerada en GPU.

    Utiliza `POT` (Python Optimal Transport) si está disponible, o un solver
    Sinkhorn entropic regularizado nativo en PyTorch en GPU.
    """
    if len(mat1) == 0 or len(mat2) == 0:
        return float("inf")

    if len(mat1) == 1 and len(mat2) == 1:
        return float(np.linalg.norm(mat1[0] - mat2[0]))

    if POT_AVAILABLE:
        D = np.linalg.norm(mat1[:, None, :] - mat2[None, :, :], axis=-1)
        a = np.ones(len(mat1)) / len(mat1)
        b = np.ones(len(mat2)) / len(mat2)
        try:
            return float(ot.emd2(a, b, D))
        except Exception:
            pass

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch no está disponible para cálculo Sinkhorn.")

    dev = device or get_torch_device("auto")
    dtype = torch.float32 if getattr(dev, "type", "") == "mps" else torch.float64
    t1 = torch.as_tensor(mat1, dtype=dtype, device=dev)
    t2 = torch.as_tensor(mat2, dtype=dtype, device=dev)
    n_a = t1.shape[0]
    n_b = t2.shape[0]

    # Matriz de costes euclidianos
    M = torch.cdist(t1, t2, p=2.0)

    # Si reg no se especifica, escala según distancias
    if reg is None:
        med = float(torch.median(M).item())
        reg = max(0.5, med * 0.2)

    # Distribuciones marginales uniformes
    a = torch.full((n_a,), 1.0 / n_a, dtype=dtype, device=dev)
    b = torch.full((n_b,), 1.0 / n_b, dtype=dtype, device=dev)

    # Algoritmo Sinkhorn-Knopp en GPU con guardias numéricas
    K = torch.exp(-M / reg)
    K = torch.clamp(K, min=1e-12)
    u = torch.ones(n_a, dtype=dtype, device=dev)

    for _ in range(max_iter):
        v = b / (K.T @ u + 1e-12)
        u = a / (K @ v + 1e-12)

    # Plan de transporte P = diag(u) K diag(v)
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    cost = torch.sum(P * M)
    return float(cost.item())


#  Cómputo Matricial Acelerado de Distancias 

def compute_distance_matrix_torch(
    bags: List[Bag],
    metric_name: str = "hausdorff",
    device: str = "auto",
) -> np.ndarray:
    """Calcula la matriz de distancias NxN completa directamente acelerada en GPU.

    Args:
        bags: Lista de objetos Bag.
        metric_name: Nombre de la métrica ('hausdorff', 'hausdorff_min', 'hausdorff_avg',
                     'cauchy_schwarz', 'mahalanobis', 'earth_movers').
        device: Dispositivo ('auto', 'cuda', 'mps', 'cpu').

    Returns:
        Matriz numpy NxN de distancias.
    """
    dev = get_torch_device(device)
    num_bags = len(bags)
    matrix = np.zeros((num_bags, num_bags), dtype=np.float64)

    if num_bags <= 1:
        return matrix

    # Mapeo a función de distancia en GPU
    metric_map = {
        "hausdorff": lambda m1, m2: hausdorff_torch(m1, m2, device=dev, mode="max"),
        "hausdorff_max": lambda m1, m2: hausdorff_torch(m1, m2, device=dev, mode="max"),
        "hausdorff_min": lambda m1, m2: hausdorff_torch(m1, m2, device=dev, mode="min"),
        "hausdorff_avg": lambda m1, m2: hausdorff_torch(m1, m2, device=dev, mode="avg"),
        "cauchy_schwarz": lambda m1, m2: cauchy_schwarz_torch(m1, m2, device=dev),
        "mahalanobis": lambda m1, m2: mahalanobis_torch(m1, m2, device=dev),
        "earth_movers": lambda m1, m2: sinkhorn_emd_torch(m1, m2, device=dev),
    }

    fn = metric_map.get(metric_name.lower())
    if fn is None:
        raise ValueError(f"Métrica '{metric_name}' no soportada en backend GPU.")

    # Extraemos matrices NumPy de las bolsas
    matrices = [b.as_matrix() for b in bags]

    logger.info(f"[GPU] Calculando matriz de distancias ({num_bags}x{num_bags}) en dispositivo {dev}...")

    for i in range(num_bags):
        m1 = matrices[i]
        for j in range(i + 1, num_bags):
            m2 = matrices[j]
            d = fn(m1, m2)
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix
