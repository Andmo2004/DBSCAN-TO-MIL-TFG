"""
tests/models/conftest.py

Fixtures y utilidades específicos para tests de modelos (clustering, classification).

Los fixtures principales se definen en tests/conftest.py (nivel raíz):
    - binary_train, binary_test: Datasets binarios bien separados (10+10 y 5+5)
    - tiny_train, tiny_test: Datasets pequeños para tests rápidos (3+3 y 2+2)
    - empty_dataset: Dataset vacío (edge case)
    - single_bag_dataset: Dataset con 1 bolsa (edge case)

Helpers:
    - _schema(n): Crea esquema con n atributos numéricos
    - _make_bag_custom(): Para crear bolsas con matrices personalizadas
    - _make_binary_dataset(): Para crear datasets binarios custom con seeds

Principios:
    - Todos los objetos se construyen en memoria (sin I/O)
    - Seed fijo para reproducibilidad
    - Positivos en cluster [2, 3], negativos en cluster [0, 1]
"""

import numpy as np
import pytest

from miclustering.data.attribute import Attribute
from miclustering.data.bag import Bag
from miclustering.data.instance import Instance
from miclustering.data.midata import MIData


# ─── Helpers privados para tests de modelos ────────────────────────────────────

def _schema(n: int = 4) -> list[Attribute]:
    """Crea esquema con n atributos numéricos."""
    return [Attribute(f"f{i}", "real") for i in range(n)]


def _make_bag_custom(bag_id: str, label: int, matrix: list[list[float]], 
                     schema: list[Attribute] | None = None) -> Bag:
    """Crea una bolsa con matriz personalizada."""
    if schema is None:
        schema = _schema(len(matrix[0]))
    insts = [Instance(list(row), schema) for row in matrix]
    return Bag(bag_id=bag_id, label=str(label), instances=insts)


def _make_binary_dataset(
    n_pos: int = 10,
    n_neg: int = 10,
    n_inst: int = 5,
    n_feat: int = 4,
    *,
    seed: int = 0,
    name: str = "synthetic",
) -> MIData:
    """Dataset binario en memoria, sin I/O.
    
    Positivos: cluster en [2, 3]
    Negativos: cluster en [0, 1]
    """
    rng = np.random.RandomState(seed)
    s = _schema(n_feat)
    bags = []
    # Positivos: cluster en [2, 3]
    for i in range(n_pos):
        mat = (rng.rand(n_inst, n_feat) + 2.0).tolist()
        bags.append(Bag(f"pos_{i}", "1", [Instance(r, s) for r in mat]))
    # Negativos: cluster en [0, 1]
    for i in range(n_neg):
        mat = rng.rand(n_inst, n_feat).tolist()
        bags.append(Bag(f"neg_{i}", "0", [Instance(r, s) for r in mat]))
    return MIData(bags, name)
