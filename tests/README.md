# tests/

Suite de tests unitarios e integración para **MIClustering**.

Todos los tests se construyen en memoria — sin archivos ARFF en disco, sin dependencias externas de I/O — lo que garantiza ejecuciones rápidas, deterministas y reproducibles en cualquier entorno.

---

## Tabla de contenidos

- [Estructura](#estructura)
- [Ejecución](#ejecución)
- [Arquitectura de tests](#arquitectura-de-tests)
- [Fixtures compartidos](#fixtures-compartidos)
- [Módulos cubiertos](#módulos-cubiertos)
- [Convenciones](#convenciones)
- [Añadir nuevos tests](#añadir-nuevos-tests)
- [Tests xfail — bugs conocidos](#tests-xfail--bugs-conocidos)
- [Cobertura](#cobertura)

---

## Estructura

```
tests/
├── conftest.py                  # Fixtures globales (schema, bag, dataset…)
│
├── data/
│   ├── __init__.py
│   ├── test_attribute_instance.py   # Attribute, Instance
│   ├── test_bag.py                  # Bag y su protocolo de secuencia
│   ├── test_midata.py               # MIData: contenedor, split, queries
│   └── test_utils_and_arff.py       # parse_label, ArffToMIData
│
├── distances/
│   ├── __init__.py
│   ├── test_hausdorff.py            # Hausdorff max/min/avg
│   └── test_probability_distribution.py  # Cauchy-Schwarz, EMD, Mahalanobis
│
├── evaluation/
│   ├── __init__.py
│   └── test_evaluation.py           # scoring, bcm (Hungarian), CVIs internos
│
├── models/
│   ├── __init__.py
│   ├── conftest.py                  # Helpers y datasets específicos de modelos
│   ├── test_models_common.py        # Contrato scikit-learn parametrizado (todos los modelos)
│   ├── test_midbscan.py             # MIDBSCAN
│   ├── test_mikmeans.py             # MIKMeans
│   ├── test_mikmedoids.py           # MIKMedoids
│   └── test_miknn.py                # MIKnn
│
├── preprocessing/
│   ├── __init__.py
│   └── test_preprocessing.py        # MinMaxScaler, StandardScaler
│
└── run/
    ├── __init__.py
    └── test_run.py                  # RunConfig, run_pipeline, run_json
```

---

## Ejecución

### Todos los tests

```bash
# Desde la raíz del proyecto
pytest
```

### Módulo específico

```bash
pytest tests/models/
pytest tests/data/test_bag.py
pytest tests/models/test_miknn.py::TestMIKnnPredict
```

### Con cobertura

```bash
pytest --cov=miclustering --cov-report=term-missing
```

### Solo tests rápidos (sin los de integración marcados como slow)

```bash
pytest -m "not slow"
```

### Ver tests xfail y sus razones

```bash
pytest -v --tb=short 2>&1 | grep -E "XFAIL|XPASS"
```

---

## Arquitectura de tests

### Principios fundamentales

**Sin I/O en tests unitarios.** Todos los objetos se construyen en memoria usando los helpers de `conftest.py`. Ningún test lee archivos ARFF del disco.

**Tests de caja negra.** Se testea comportamiento observable (valores de retorno, excepciones, tipos), no detalles de implementación como atributos privados.

**Fixtures compuestos.** Las fixtures siguen la jerarquía del dominio: `Attribute → Instance → Bag → MIData`. Las más simples se componen para formar las más complejas.

**Aislamiento total.** `scope="function"` por defecto: cada test recibe objetos frescos. No hay estado compartido entre tests.

**Matrices precomputadas para modelos.** Los tests de `MIDBSCAN` y `MIKMedoids` usan `precomputed_matrix` sintética para desacoplar la lógica de clustering de las funciones de distancia (ya cubiertas en `tests/distances/`).

### Separación contrato / comportamiento

```
tests/models/test_models_common.py   ← QUÉ hace la interfaz (contrato scikit-learn)
tests/models/test_midbscan.py        ← CÓMO lo hace MIDBSCAN (comportamiento específico)
tests/models/test_mikmeans.py        ← CÓMO lo hace MIKMeans
...
```

El archivo `test_models_common.py` parametriza los mismos tests sobre los cuatro modelos usando `pytest.mark.parametrize`, evitando duplicación y garantizando que todos implementan el mismo contrato público.

---

## Fixtures compartidos

Definidos en `tests/conftest.py` y disponibles en toda la suite sin importación explícita.

### Primitivos

| Fixture | Tipo | Descripción |
|---|---|---|
| `real_attribute` | `Attribute` | Atributo numérico `"real"` |
| `nominal_attribute` | `Attribute` | Atributo nominal `{neg, pos}` |
| `schema_3f` | `List[Attribute]` | Esquema de 3 atributos reales |
| `basic_instance` | `Instance` | Instancia con valores `[1.0, 2.0, 3.0]` |

### Bolsas

| Fixture | Tipo | Descripción |
|---|---|---|
| `basic_bag` | `Bag` | 3 instancias, label=1, bag_id="bag_A" |
| `empty_bag` | `Bag` | Sin instancias, label=0 |
| `singleton_bag` | `Bag` | 1 instancia |

### Datasets

| Fixture | Tipo | Descripción |
|---|---|---|
| `small_dataset` | `MIData` | 6 bolsas sintéticas (3 pos + 3 neg) |
| `binary_dataset_10` | `MIData` | 10 bolsas para tests de split |
| `binary_train` | `MIData` | 20 bolsas bien separadas (10+10) para modelos |
| `binary_test` | `MIData` | 10 bolsas para evaluación de modelos |
| `tiny_train` / `tiny_test` | `MIData` | Versiones reducidas para tests rápidos |
| `empty_dataset` | `MIData` | Dataset vacío (edge cases) |
| `single_bag_dataset` | `MIData` | Dataset con 1 bolsa (edge cases) |

### Helpers

```python
from tests.conftest import make_schema, make_instance, make_bag, make_dataset

schema = make_schema(n_features=4)
bag = make_bag(bag_id="b0", label=1, n_instances=3, n_features=4)
dataset = make_dataset(n_bags=10, n_instances=5, n_features=4, seed=42)
```

Los helpers de `tests/models/conftest.py` añaden:

```python
from tests.models.conftest import _make_binary_dataset, _schema

# Dataset binario con clusters bien separados (positivos en [2,3], negativos en [0,1])
train = _make_binary_dataset(n_pos=10, n_neg=10, seed=42)
```

---

## Módulos cubiertos

### `tests/data/`

Cubre la capa de dominio del proyecto.

- **`test_attribute_instance.py`** — construcción, propiedades, validación de tipos, `__slots__`, igualdad estructural de `Attribute` e `Instance`.
- **`test_bag.py`** — protocolo de contenedor/secuencia (`len`, `iter`, `[]`, `in`), `add_instance`, `as_matrix` (contrato NumPy crítico para distancias), igualdad, representaciones.
- **`test_midata.py`** — almacenamiento, `split_data` (reproducibilidad, proporciones, sin solapamiento), `get_positive_bags` / `get_negative_bags`, protocolo de secuencia, igualdad.
- **`test_utils_and_arff.py`** — `parse_label` con enteros, flotantes, strings numéricos, strings nominales (mapa por defecto y custom), bytes. `ArffToMIData` para errores de I/O y propiedades del loader.

### `tests/distances/`

Verifica propiedades matemáticas de cada métrica con cálculos manuales documentados en los docstrings.

- **`test_hausdorff.py`** — las tres variantes (max, min, avg): identidad, simetría, no-negatividad, bolsas vacías → `inf`, invariante `d_min ≤ d_avg ≤ d_max`, casos manuales con valores esperados exactos, robustez numérica (1D, valores grandes, bolsas casi idénticas).
- **`test_probability_distribution.py`** — Cauchy-Schwarz (ortogonal, opuesto, paralelo, ángulo 45°, multi-instancia), EMD (1×1, 2×2 alineado, tamaños asimétricos), Mahalanobis (covarianza singular → pseudoinversa, isótropo, alta dimensión).

### `tests/evaluation/`

- **`test_evaluation.py`** — `detect_imbalance_ratio` (balance, desbalance, clase única), `score_labels` (todo ruido → 0, 1 cluster → bajo, 2 clusters perfectos → alto, penalización por ruido excesivo), mapeo húngaro (`MILEvaluator`), todos los CVIs internos (SED, DD, Hc, VRC, I) e `InternalCVIEvaluator` (registro, chaining, verbose).

### `tests/models/`

- **`test_models_common.py`** — contrato scikit-learn parametrizado sobre los 4 modelos: estado inicial, `fit()`, `predict()`, `fit_predict()`, `get_statistics()`, inmutabilidad de `labels`, representaciones.
- **`test_midbscan.py`** — matrices sintéticas `_identity_matrix` / `_block_matrix`, todo-ruido con `eps` pequeño, 2 clusters con `_block_matrix`, `precomputed_matrix` de forma incorrecta → `ValueError`, `get_noise_points`, `get_cluster_members`.
- **`test_mikmeans.py`** — reproducibilidad con `random_state`, centroides como objetos `Bag` válidos, convergencia, `k > n_bags` → ajuste automático.
- **`test_mikmedoids.py`** — medoides son bolsas reales del training set, matriz liberada post-`fit` (`_distance_matrix is None`), 2 clusters con `_block_matrix`.
- **`test_miknn.py`** — dataset perfectamente separable para tests deterministas, `predict_proba` suma a 1.0, `get_neighbors` ordenados por distancia, `predict_bag` consistente con `predict`.

### `tests/preprocessing/`

- **`test_preprocessing.py`** — contrato `BaseScaler` parametrizado, `MinMaxScaler` (valores exactos min/max/intermedios, rango custom, `inverse_transform` round-trip), `StandardScaler` (μ≈0, σ≈1 post-transform, round-trip), edge cases (feature constante, solo nominales, valores grandes).

### `tests/run/`

- **`test_run.py`** — `RunConfig.from_dict` con aliases en español, validaciones de campos, `run_pipeline` con los 4 algoritmos y 3 métricas de distancia, `run_json` con datos en memoria y guardado de resultados.

---

## Convenciones

### Nomenclatura

```
tests/<modulo>/test_<archivo_fuente>.py
```

Las clases de test agrupan por responsabilidad:

```python
class TestMIKnnConstruction:    # tests de __init__ y validaciones
class TestMIKnnFit:             # tests de fit()
class TestMIKnnPredict:         # tests de predict()
```

Los nombres de test describen qué se verifica:

```python
def test_fit_empty_dataset_raises_value_error(self):
def test_predict_returns_dict(self):
def test_k_zero_raises(self):
```

### Aserciones

Se usa `pytest.approx` para comparaciones de punto flotante:

```python
assert result == pytest.approx(1.0)
assert value == pytest.approx(expected, rel=1e-6)
```

Para arrays NumPy:

```python
np.testing.assert_allclose(result, expected, atol=1e-10)
np.testing.assert_array_equal(result, expected)
```

### Matrices precomputadas en tests de modelos

Para evitar el coste de calcular distancias reales en tests unitarios:

```python
# Todos los puntos cerca → un único cluster con eps suficiente
def _identity_matrix(n): return np.zeros((n, n))

# Dos grupos bien separados → exactamente 2 clusters
def _block_matrix(n_a, n_b, intra=0.1, inter=10.0): ...
```

---

## Añadir nuevos tests

### Test unitario de una nueva función

```python
# tests/data/test_mi_nueva_clase.py
import pytest
from miclustering.data.mi_nueva_clase import MiNuevaClase

class TestMiNuevaClaseConstruction:

    def test_stores_parameter(self):
        obj = MiNuevaClase(param=42)
        assert obj.param == 42

    def test_invalid_param_raises_value_error(self):
        with pytest.raises(ValueError, match="param"):
            MiNuevaClase(param=-1)
```

### Test de un nuevo modelo

Los nuevos modelos deben añadirse al parametrize de `test_models_common.py`:

```python
# En tests/models/test_models_common.py
from miclustering.models.mi_nuevo_modelo import MiNuevoModelo

MODEL_FACTORIES = [
    # ... modelos existentes ...
    pytest.param(
        lambda: MiNuevoModelo(param=2),
        id="MiNuevoModelo",
    ),
]
```

Y crear su propio archivo de tests de comportamiento:

```python
# tests/models/test_mi_nuevo_modelo.py
class TestMiNuevoModeloConstruction:    ...
class TestMiNuevoModeloFit:             ...
class TestMiNuevoModeloPredict:         ...
```

### Usar fixtures existentes

```python
# Cualquier test puede usar las fixtures del conftest raíz sin importar nada
def test_algo(self, basic_bag, schema_3f):
    ...

# Fixtures de modelos requieren estar en tests/models/ o importar explícitamente
def test_algo(self):
    ds = _make_binary_dataset(n_pos=5, n_neg=5, seed=0)
```

---

## Tests xfail — bugs conocidos

Los tests marcados con `@pytest.mark.xfail` documentan bugs o limitaciones de diseño conocidas. Se muestran en el reporte de pytest pero no bloquean CI.

| Test | Motivo |
|---|---|
| `TestMIKnnKnownIssues::test_nominal_string_labels_parsed_correctly` | `MIKnn._parse_label` no delega a `utils.parse_label`, no acepta etiquetas `"positive"/"negative"` |
| `TestArffToMIDataProperties::test_validate_structure_can_be_tested_without_file` | `ArffToMIData.load()` mezcla I/O y validación en un único método; dificulta testear `_validate_structure` de forma aislada |
| `TestScalerDesignAuditNotes::test_transform_does_not_access_private_instance_values` | `BaseScaler._create_transformed_dataset` accede a `instance._values` directamente (atributo privado con `__slots__`) |

Para ver el estado actual de todos los xfail:

```bash
pytest -v --tb=no -q 2>&1 | grep -E "XFAIL|XPASS|xfailed"
```

---

## Cobertura

Para generar un reporte HTML detallado:

```bash
pytest --cov=miclustering --cov-report=html --cov-report=term-missing
# Abre htmlcov/index.html
```

Para ver únicamente las líneas no cubiertas por módulo:

```bash
pytest --cov=miclustering --cov-report=term-missing 2>&1 | grep -v "100%"
```

Los módulos con mayor criticidad y cobertura esperada ≥ 90 %:

- `miclustering.data.*` — dominio central
- `miclustering.distances.*` — métricas
- `miclustering.models.*` — algoritmos
- `miclustering.preprocessing.scaler` — escaladores
- `miclustering.evaluation.*` — métricas de evaluación