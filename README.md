# MIClustering

> Librería de clustering y clasificación para **Multi-Instance Learning (MIL)** en Python.

MIClustering implementa algoritmos adaptados al paradigma MIL, donde los datos se organizan en *bolsas* (bags) que contienen múltiples instancias. Incluye distancias especializadas entre bolsas, preprocesado, evaluación interna y externa, pipeline de experimentación configurable mediante JSON y caché persistente de matrices de distancias.

---

## Tabla de contenidos

- [Instalación](#instalación)
- [Inicio rápido](#inicio-rápido)
- [Ejecución desde JSON](#ejecución-desde-json-run_json)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Modelos](#modelos)
- [Métricas de distancia](#métricas-de-distancia)
- [Preprocesado](#preprocesado)
- [Evaluación](#evaluación)
- [Referencia de la API](#referencia-de-la-api)
- [Tests](#tests)

---

## Instalación

**Requisito previo:** Python ≥ 3.8

### Con `uv` (recomendado)

```bash
# Instalar uv
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crear entorno e instalar dependencias
uv venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

uv pip install -e .          # instala la librería en modo editable con sus deps
```

### Con pip

```bash
pip install miclustering @ git+https://github.com/Andmo2004/MIClustering.git
```

### Desde el código fuente

```bash
git clone https://github.com/Andmo2004/MIClustering.git
cd MIClustering
pip install -e .
```

---

## Inicio rápido

```python
from miclustering import MIData, MIDBSCAN, MIKMeans, MIKMedoids, MIKnn
from miclustering.preprocessing.scaler import MinMaxScaler
from miclustering.evaluation.bcm import MILEvaluator

# 1. Cargar dataset desde ARFF
dataset = MIData.from_arff("datasets/musk1.arff")
train_data, test_data = dataset.split_data(percentage_train=70, seed=42)

# 2. Normalizar
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled  = scaler.transform(test_data)

# 3. Entrenar con MIDBSCAN
model = MIDBSCAN(epsilon=0.5, min_pts=2, metric="hausdorff_avg")
model.fit(train_scaled)

# 4. Predecir y evaluar
predictions = model.predict(test_scaled)
MILEvaluator.evaluate(test_scaled, predictions, title="MIDBSCAN — musk1")
```

Salida esperada:

```
============================================================
REPORTE DE CLASIFICACIÓN: MIDBSCAN — musk1
============================================================
Métricas:
Métrica         | Valor
------------------------------
Precision       | 0.8421
Recall          | 0.7619
F1-Score        | 0.8000
Specificity     | 0.8750
```

---

## Ejecución desde JSON (`run_json`)

Para ejecutar experimentos de forma programática y reproducible, MIClustering proporciona la función `run_json`, que lee una configuración desde un archivo JSON y ejecuta automáticamente el pipeline completo (carga de datos, escalado, entrenamiento, evaluación).

### Formato del archivo de configuración JSON

```json
{
  "dataset": "musk1",
  "medida_de_distancia": "hausdorff",
  "metodo_de_escalado": "MinMaxScaler",
  "semilla": 42,
  "algoritmo": "midbscan",
  "hiperparametros": {
    "epsilon": 0.5,
    "min_pts": 2
  },
  "optimizar_optuna": false
}
```

#### Campos disponibles

| Campo | Alias español | Tipo | Valores posibles | Default |
|---|---|---|---|---|
| `dataset` | — | string | Nombre del dataset (ej: `"musk1"`) | **requerido** |
| `distance_metric` | `medida_de_distancia` | string | `hausdorff`, `hausdorff_avg`, `hausdorff_min`, `earth_movers`, `mahalanobis`, `cauchy_schwarz` | `"hausdorff"` |
| `scaler` | `metodo_de_escalado` | string\|null | `"MinMaxScaler"`, `"StandardScaler"`, `"none"` | `"MinMaxScaler"` |
| `seed` | `semilla` | integer | Cualquier entero positivo | `42` |
| `algorithm` | `algoritmo` | string | `"midbscan"`, `"mikmeans"`, `"mikmedoids"`, `"miknn"` | `"midbscan"` |
| `hyperparams` | `hiperparametros` | object | Depende del algoritmo | `{}` |
| `optuna_optimize` | `optimizar_optuna` | boolean | `true` / `false` | `false` |
| `n_trials` | `optuna_trials` | integer | Número de trials | `30` |
| `train_pct` | `porcentaje_entrenamiento` | float | 0–100 | `70.0` |

#### Hiperparámetros por algoritmo

| Algoritmo | Parámetros |
|---|---|
| **MIDBSCAN** | `epsilon` (float), `min_pts` (int) |
| **MIKMeans** | `k` (int), `max_iters` (int) |
| **MIKMedoids** | `k` (int), `max_iters` (int) |
| **MIKnn** | `k` (int) |

### Uso básico

```python
from miclustering.run import run_json

result = run_json("config.json", verbose=True)

print(result["metrics"]["F1-Score"])   # 0.84
print(result["hyperparams"])           # {"epsilon": 0.5, "min_pts": 2}
print(result["config"]["dataset"])     # "musk1"
```

### Guardado automático de resultados

```python
result = run_json(
    "config.json",
    output_dir="results/",
    verbose=True
)
# Se guarda en: results/run_midbscan_musk1_<timestamp>.json
print(result["output_file"])
```

### Uso programático con datos en memoria

```python
from miclustering.run import run_json
from miclustering import MIData

dataset = MIData.from_arff("datasets/musk1.arff")
train_data, test_data = dataset.split_data(percentage_train=70, seed=42)

# Ejecutar múltiples configuraciones sin releer del disco
result1 = run_json("config1.json", train_data=train_data, test_data=test_data)
result2 = run_json("config2.json", train_data=train_data, test_data=test_data)
```

### Uso de `run_pipeline` directamente

```python
from miclustering.run import run_pipeline, RunConfig

config = RunConfig.from_dict({
    "dataset": "musk1",
    "algorithm": "miknn",
    "hiperparametros": {"k": 3},
    "medida_de_distancia": "hausdorff",
})
result = run_pipeline(train_data, test_data, config)
```

### Estructura del resultado

```python
{
    "config":      { ... },           # dict original del JSON
    "metrics": {
        "Accuracy":    0.8421,
        "Precision":   0.8421,
        "Recall":      0.7619,
        "F1-Score":    0.8000,
        "F1-Macro":    0.7950,
        "Specificity": 0.8750,
    },
    "model_stats": { ... },           # salida de model.get_statistics()
    "hyperparams": {"epsilon": 0.5, "min_pts": 2},
    "mapping":     {0: 1, 1: 0},      # cluster→clase (vacío para MIKnn)
    "output_file": "results/run_midbscan_musk1_20260528_143045.json",
}
```

### Optimización de hiperparámetros (Optuna)

```json
{
  "dataset": "musk1",
  "algoritmo": "midbscan",
  "optimizar_optuna": true,
  "optuna_trials": 30
}
```

Requiere `pip install optuna`.

---

## Estructura del proyecto

```
MIClustering/
├── src/
│   └── miclustering/
│       ├── __init__.py         # API pública: MIDBSCAN, MIKMeans, MIKMedoids, MIKnn, MIData, Bag
│       ├── data/
│       │   ├── attribute.py           # Descriptor de columna (inmutable, __slots__)
│       │   ├── instance.py            # Instancia individual con validación de tipos
│       │   ├── bag.py                 # Bolsa: contenedor de instancias con protocolo de secuencia
│       │   ├── midata.py              # Dataset MIL: contenedor de bolsas con split, queries
│       │   ├── arff_reader.py         # Lector ARFF → MIData (scipy.io.arff)
│       │   └── utils.py               # parse_label: int/float/str/bytes/nominal → int
│       ├── distances/
│       │   ├── __init__.py            # DISTANCE_REGISTRY: dict nombre → función
│       │   ├── hausdorff.py           # Hausdorff max, min, avg
│       │   ├── probability_distribution.py  # Cauchy-Schwarz, EMD (LP), Mahalanobis
│       │   ├── distance_matrix.py     # Cálculo matricial simétrico (N×N)
│       │   └── matrix_cache.py        # Caché LRU persistente en disco (.npy)
│       ├── models/
│       │   ├── midbscan.py            # DBSCAN para MIL
│       │   ├── mikmeans.py            # K-Means para MIL (centroides sintéticos)
│       │   ├── mikmedoids.py          # K-Medoids PAM para MIL
│       │   └── miknn.py               # k-NN supervisado para MIL
│       ├── preprocessing/
│       │   └── scaler.py              # BaseScaler, MinMaxScaler, StandardScaler
│       ├── evaluation/
│       │   ├── bcm.py                 # MILEvaluator: Hungarian mapping + métricas BCM
│       │   ├── cvi.py                 # CVIs internos: SED, DD, Hc, VRC, I
│       │   └── scoring.py             # score_labels, detect_imbalance_ratio (Optuna)
│       └── run/
│           ├── __init__.py            # Exporta: run_json, RunConfig, run_pipeline
│           ├── _config.py             # RunConfig: dataclass validada desde JSON
│           ├── _pipeline.py           # run_pipeline: lógica pura sin I/O
│           └── json_runner.py         # run_json: entrada pública con I/O
├── datasets/                          # Archivos ARFF (no incluidos en el repo)
├── tests/                             # Suite de tests unitarios (ver tests/README.md)
├── pyproject.toml
└── README.md
```

---

## Modelos

Todos los modelos heredan de `BaseEstimator` / `ClusterMixin` o `ClassifierMixin` (scikit-learn) e implementan la interfaz común:

```
fit(dataset)                → self
predict(dataset)            → Dict[str, int]
fit_predict(X, y=None)      → Dict[str, int]
get_statistics()            → Dict[str, Any]
labels                      → Dict[str, int]  (copia, inmutable)
is_fitted                   → bool
```

### MIDBSCAN

Algoritmo DBSCAN adaptado a MIL. Detecta clusters de densidad arbitraria y marca ruido sin necesidad de especificar el número de clusters.

```python
from miclustering import MIDBSCAN

model = MIDBSCAN(
    epsilon=0.5,        # radio de vecindad
    min_pts=2,          # mínimo de bolsas para ser punto núcleo
    metric="hausdorff"  # métrica de distancia entre bolsas
)
model.fit(train_data)
labels = model.labels           # Dict[bag_id, cluster_id]; ruido → -1
stats  = model.get_statistics()

# Con matriz precomputada (evita recalcular distancias)
from miclustering.distances.distance_matrix import compute_distance_matrix
from miclustering.distances import DISTANCE_REGISTRY

dist_matrix = compute_distance_matrix(
    train_data.bags, DISTANCE_REGISTRY["hausdorff"], "hausdorff"
)
model.fit(train_data, precomputed_matrix=dist_matrix)
```

### MIKMeans

K-Means adaptado a MIL. Los centroides son bolsas sintéticas construidas como la media de todas las instancias de las bolsas asignadas al cluster.

```python
from miclustering import MIKMeans

model = MIKMeans(k=2, metric="hausdorff_avg", max_iters=100, random_state=42)
model.fit(train_data)
predictions = model.predict(test_data)
print(model.centroids)   # Lista de Bag sintéticos (un centroide por cluster)
```

### MIKMedoids

K-Medoids (algoritmo PAM) adaptado a MIL. A diferencia de K-Means, los medoides son bolsas reales del dataset, lo que lo hace más robusto frente a outliers.

```python
from miclustering import MIKMedoids

model = MIKMedoids(k=2, metric="hausdorff_min", random_state=42)
model.fit(train_data, precomputed_matrix=dist_matrix)  # acepta matriz precomputada
print(model.medoids)   # Lista de bolsas reales que actúan como medoides
```

### MIKnn

Clasificador k-Nearest Neighbors para MIL. Lazy learning: almacena el conjunto de entrenamiento y clasifica por mayoría de votos con desempate por distancia acumulada. Es el único modelo supervisado de la librería — `predict()` devuelve directamente etiquetas de clase (0/1).

```python
from miclustering import MIKnn

model = MIKnn(k=3, metric="hausdorff")
model.fit(train_data)

predictions  = model.predict(test_data)
probabilities = model.predict_proba(test_data)   # {bag_id: {0: p0, 1: p1}}
single_pred  = model.predict_bag(test_data.bags[0])
neighbors    = model.get_neighbors(test_data.bags[0])  # [(bag_id, label, dist), ...]
```

---

## Métricas de distancia

| Nombre | Descripción |
|---|---|
| `hausdorff` | Hausdorff máxima (simétrica) — penaliza el peor caso |
| `hausdorff_min` | Mínimo absoluto entre instancias — sensible a instancias cercanas |
| `hausdorff_avg` | Hausdorff promedio normalizada por `\|A\| + \|B\|` |
| `earth_movers` | Earth Mover's Distance (LP) — transporte óptimo entre distribuciones |
| `mahalanobis` | Distancia de Mahalanobis entre distribuciones gaussianas |
| `cauchy_schwarz` | Similitud coseno entre centroides de bolsa; rango [0, 2] |

Todas las métricas aceptan objetos `Bag` y devuelven `float`. Disponibles en `DISTANCE_REGISTRY`:

```python
from miclustering.distances import DISTANCE_REGISTRY

dist_func = DISTANCE_REGISTRY["hausdorff_avg"]
d = dist_func(bag_a, bag_b)   # float
```

### Cálculo manual de una matriz de distancias

```python
from miclustering.distances.distance_matrix import compute_distance_matrix
from miclustering.distances.hausdorff import hausdorff_distance_avg

matrix = compute_distance_matrix(
    bags=train_data.bags,
    metric_func=hausdorff_distance_avg,
    metric_name="hausdorff_avg"
)
# → np.ndarray (N × N), simétrica, diagonal 0
```

### Caché persistente

```python
from miclustering.distances.matrix_cache import global_persistent_cache
from miclustering.distances.hausdorff import hausdorff_distance_avg

matrix = global_persistent_cache.get(
    dataset_name="musk1",
    split="train",
    scaler_name="minmax",
    metric_name="hausdorff_avg",
    bags=train_data.bags,
    metric_func=hausdorff_distance_avg,
    save=True,   # persiste en .miclustering_cache/distance_matrices/
)
```

La caché guarda los archivos en `.miclustering_cache/distance_matrices/`. La ubicación puede sobreescribirse con la variable de entorno `MICLUSTERING_CACHE_DIR`.

---

## Preprocesado

### MinMaxScaler

Escala todos los atributos numéricos al rango `[0, 1]` (configurable). Los atributos nominales y de cadena se conservan sin cambios.

```python
from miclustering.preprocessing.scaler import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled  = scaler.transform(test_data)

# Revertir la transformación
original = scaler.inverse_transform(test_scaled)
```

### StandardScaler

Estandariza los atributos numéricos a media 0 y desviación estándar 1.

```python
from miclustering.preprocessing.scaler import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled  = scaler.transform(test_data)
```

Ambos scalers:
- Solo actúan sobre atributos de tipo `real` o `integer`.
- Siguen la convención scikit-learn: `fit` en train, `transform` en train y test.
- `fit_transform` equivale a `fit` + `transform` en un único paso.
- `transform` devuelve un nuevo `MIData` sin mutar el original (`inplace=False` por defecto).

---

## Evaluación

### Métricas de clasificación binaria (BCM)

Compara las asignaciones de cluster contra las etiquetas reales usando el **algoritmo Húngaro** para el mapeo óptimo cluster → clase.

```python
from miclustering.evaluation.bcm import MILEvaluator

results = MILEvaluator.evaluate(
    dataset=test_data,
    model_labels=predictions,   # Dict[bag_id, cluster_id]
    title="Experimento 1"
)
# results → {"Precision": 0.84, "Recall": 0.76, "F1-Score": 0.80, "Specificity": 0.87}

# Mapeo directo sin impresión
y_pred_mapped, mapping = MILEvaluator.hungarian_map_clusters_to_labels(y_true, y_pred_raw)
```

### Índices de validación interna (CVI)

Evalúan la calidad del clustering sin usar etiquetas, útiles para selección de hiperparámetros.

| Índice | Tipo | Criterio | Requiere X |
|---|---|---|---|
| `SED` | Compactibilidad | ↓ menor es mejor | Sí |
| `DD` | Compactibilidad | ↓ menor es mejor | Sí |
| `Hc` | Compactibilidad (entropía) | ↓ menor es mejor | No |
| `VRC` | Compact. + Separación | ↑ mayor es mejor | Sí |
| `I` (PBM) | Compact. + Separación | ↑ mayor es mejor | Sí |

```python
from miclustering.evaluation.cvi import InternalCVIEvaluator

evaluator = InternalCVIEvaluator()
results = evaluator.evaluate(
    dist_matrix=distance_matrix,
    labels=model.labels,
    bag_ids=[bag.bag_id for bag in train_scaled.bags],
    dataset=train_scaled,        # necesario para SED, DD, VRC, I
    title="MIDBSCAN eps=0.5"
)

# Añadir CVIs específicos
from miclustering.evaluation.cvi import VRCIndex, IIndex
evaluator_custom = InternalCVIEvaluator(cvis=[VRCIndex(), IIndex()])
```

---

## Referencia de la API

### `MIData`

```python
MIData(bags: List[Bag], name: str)
MIData.from_arff(file_path, dataset_name=None, bag_column="bag", class_column="class")

dataset.get_bag(i)                          # → Bag
dataset.get_num_bags()                      # → int
dataset.split_data(percentage_train, seed)  # → (MIData, MIData)
dataset.get_labels()                        # → List
dataset.get_positive_bags()                 # → List[Bag]
dataset.get_negative_bags()                 # → List[Bag]
dataset.bags                                # → List[Bag] (copia)
```

### `Bag`

```python
Bag(bag_id, label, instances=None)

bag.get_instance(i)       # → Instance
bag.get_num_instances()   # → int
bag.add_instance(inst)
bag.as_matrix()           # → np.ndarray (n_instances × n_features), float64
bag.bag_id                # → Any
bag.label                 # → Any (setter disponible)
bag.instances             # → List[Instance] (copia)
```

### `parse_label`

```python
from miclustering.data.utils import parse_label

parse_label(1)           # → 1
parse_label("1.0")       # → 1
parse_label("positive")  # → 1  (mapa por defecto)
parse_label(b"0")        # → 0
parse_label("musk", nominal_map={"musk": 1, "non_musk": 0})  # → 1
```

---

## Tests

La suite de tests cubre los módulos `data`, `distances`, `evaluation`, `models`, `preprocessing` y `run`. Todos los tests se construyen en memoria — sin archivos ARFF, sin dependencias externas de I/O.

```bash
# Ejecutar todos los tests
pytest

# Con cobertura
pytest --cov=miclustering --cov-report=term-missing

# Módulo específico
pytest tests/models/
```

Ver [`tests/README.md`](tests/README.md) para documentación completa de la suite: fixtures disponibles, convenciones, cómo añadir nuevos tests y bugs conocidos documentados como `xfail`.
