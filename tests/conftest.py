"""
tests/conftest.py

Fixtures compartidos por toda la suite de tests del módulo `data`.
Centralizar fixtures aquí evita duplicación y hace que los tests sean
independientes de cualquier fichero externo (ARFF, disco, etc.).

Principios aplicados:
- Todos los objetos son construidos en memoria → sin I/O en tests unitarios.
- Los fixtures de mayor granularidad se componen de los más pequeños
  (attribute → instance → bag → dataset), siguiendo la jerarquía del dominio.
- scope="function" por defecto: cada test recibe objetos frescos y aislados.
"""

import pytest
import numpy as np

from miclustering.data.attribute import Attribute
from miclustering.data.instance import Instance
from miclustering.data.bag import Bag
from miclustering.data.midata import MIData

# Helpers

def make_schema(n_features: int = 3) -> list[Attribute]:
    """Devuelve un esquema de *n_features* atributos numéricos reales."""
    return [Attribute(f"f{i}", "real") for i in range(n_features)]


def make_instance(values: list[float], schema: list[Attribute] | None = None) -> Instance:
    """Crea una instancia con los valores dados (schema inferido si no se pasa)."""
    if schema is None:
        schema = make_schema(len(values))
    return Instance(values, schema)


def make_bag(
    bag_id: str = "bag_0",
    label: int = 0,
    n_instances: int = 3,
    n_features: int = 3,
    values_matrix: list[list[float]] | None = None,
) -> Bag:
    """
    Crea una bolsa con instancias sintéticas.

    Si se pasa *values_matrix*, se usa directamente (forma n_instances × n_features).
    En caso contrario se genera con np.ones.
    """
    schema = make_schema(n_features)
    if values_matrix is None:
        values_matrix = np.ones((n_instances, n_features)).tolist()
    instances = [Instance(row, schema) for row in values_matrix]
    return Bag(bag_id=bag_id, label=label, instances=instances)


def make_dataset(
    n_bags: int = 6,
    n_instances: int = 4,
    n_features: int = 3,
    *,
    seed: int = 0,
) -> MIData:
    """
    Crea un MIData completamente sintético con n_bags bolsas.

    - La mitad de las bolsas tienen label=0 y la otra mitad label=1.
    - Las instancias se generan con np.random para variar los valores.
    """
    rng = np.random.default_rng(seed)
    schema = make_schema(n_features)
    bags = []
    for i in range(n_bags):
        label = i % 2  # alternado: 0, 1, 0, 1 …
        values_matrix = rng.random((n_instances, n_features)).tolist()
        instances = [Instance(row, schema) for row in values_matrix]
        bags.append(Bag(bag_id=f"bag_{i}", label=label, instances=instances))
    return MIData(bags, name="synthetic")


# Fixtures 

@pytest.fixture()
def real_attribute() -> Attribute:
    return Attribute("feature_0", "real")


@pytest.fixture()
def nominal_attribute() -> Attribute:
    return Attribute("class", "nominal", values=["neg", "pos"])


@pytest.fixture()
def schema_3f() -> list[Attribute]:
    return make_schema(3)


@pytest.fixture()
def basic_instance(schema_3f) -> Instance:
    return Instance([1.0, 2.0, 3.0], schema_3f)


@pytest.fixture()
def zero_instance(schema_3f) -> Instance:
    return Instance([0.0, 0.0, 0.0], schema_3f)


@pytest.fixture()
def basic_bag(schema_3f) -> Bag:
    """Bolsa con 3 instancias distintas."""
    instances = [
        Instance([1.0, 2.0, 3.0], schema_3f),
        Instance([4.0, 5.0, 6.0], schema_3f),
        Instance([7.0, 8.0, 9.0], schema_3f),
    ]
    return Bag(bag_id="bag_A", label=1, instances=instances)


@pytest.fixture()
def empty_bag() -> Bag:
    return Bag(bag_id="empty", label=0, instances=[])


@pytest.fixture()
def singleton_bag(schema_3f) -> Bag:
    return Bag(bag_id="singleton", label=0, instances=[Instance([1.0, 1.0, 1.0], schema_3f)])


@pytest.fixture()
def small_dataset() -> MIData:
    """Dataset de 6 bolsas, 3 pos + 3 neg, para tests rápidos."""
    return make_dataset(n_bags=6, n_instances=4, n_features=3, seed=42)


@pytest.fixture()
def binary_dataset_10() -> MIData:
    """Dataset de 10 bolsas para tests de split y label queries."""
    return make_dataset(n_bags=10, n_instances=5, n_features=4, seed=7)