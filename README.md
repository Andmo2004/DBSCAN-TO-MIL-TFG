# MIClustering

> Librería de clustering para **Multi-Instance Learning (MIL)** en Python.

MIClustering implementa algoritmos de clustering y clasificación adaptados al paradigma MIL, donde los datos se organizan en *bolsas* (bags) que contienen múltiples instancias. Incluye distancias especializadas entre bolsas, preprocesado, evaluación interna y externa, y caché persistente de matrices de distancias.

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
- [Contribuir](#contribuir)
- [Licencia](#licencia)

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
.venv\Scripts\activate    # Windows


uv pip install -e .          # instala la librería en modo editable con sus deps
```

### Con pip

```bash
pip install miclustering
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

#### Campos disponibles:

| Campo | Alias español | Tipo | Valores posibles | Default |
|---|---|---|---|---|
| `dataset` | — | string | Nombre del dataset (ej: `"musk1"`) | **requerido** |
| `distance_metric` | `medida_de_distancia` | string | `hausdorff`, `hausdorff_avg`, `hausdorff_min`, `earth_movers`, `mahalanobis`, `cauchy_schwarz` | `"hausdorff"` |
| `scaler` | `metodo_de_escalado` | string\|null | `"MinMaxScaler"`, `"StandardScaler"`, `"none"` (sin escalado) | `"MinMaxScaler"` |
| `seed` | `semilla` | integer | Cualquier entero positivo | `None` |
| `algorithm` | `algoritmo` | string | `"midbscan"`, `"mikmeans"`, `"mikmedoids"`, `"miknn"` | `"midbscan"` |
| `hyperparams` | `hiperparametros` | object | Depende del algoritmo | `{}` |
| `optuna_optimize` | `optimizar_optuna` | boolean | `true` / `false` | `false` |

#### Hiperparámetros por algoritmo:

**MIDBSCAN:** `epsilon` (float), `min_pts` (int)  
**MIKMeans:** `k` (int), `max_iters` (int)  
**MIKMedoids:** `k` (int)  
**MIKnn:** `k` (int)  

### Uso básico

```python
from miclustering.run import run_json

# Ejecutar desde archivo JSON
result = run_json("config.json", verbose=True)

# Acceder a los resultados
print(result["metrics"]["F1-Score"])          # 0.84
print(result["metrics"]["Precision"])         # 0.85
print(result["hyperparams"])                  # {"epsilon": 0.5, "min_pts": 2}
print(result["config"]["dataset"])            # "musk1"
```

### Guardado automático de resultados

```python
from miclustering.run import run_json

result = run_json(
    "config.json",
    output_dir="results/",
    verbose=True
)

# Se guarda automáticamente en: results/experiment_<timestamp>.json
print(result["output_file"])  # Ruta del archivo JSON guardado
```

### Uso programático con datos en memoria

Útil en notebooks o para evitar leer desde disco repetidamente:

```python
from miclustering.run import run_json
from miclustering import MIData

# Cargar datos una sola vez
dataset = MIData.from_arff("datasets/musk1.arff")
train_data, test_data = dataset.split_data(percentage_train=70, seed=42)

# Ejecutar múltiples configuraciones sin releer del disco
result1 = run_json("config1.json", train_data=train_data, test_data=test_data)
result2 = run_json("config2.json", train_data=train_data, test_data=test_data)
result3 = run_json("config3.json", train_data=train_data, test_data=test_data)
```

### Estructura del resultado

```python
{
  "dataset": "musk1",
  "config": {
    "dataset": "musk1",
    "algorithm": "midbscan",
    "distance_metric": "hausdorff",
    "scaler": "MinMaxScaler",
    "seed": 42
  },
  "hyperparams": {"epsilon": 0.5, "min_pts": 2},
  "metrics": {
    "Accuracy": 0.8421,
    "Precision": 0.8421,
    "Recall": 0.7619,
    "F1-Score": 0.8000,
    "Specificity": 0.8750
  },
  "output_file": "results/experiment_2026-05-23_14-30-45.json",
  "mapping": {0: 1, 1: 0}  # Mapeo óptimo cluster → clase (solo MIDBSCAN, MIKMeans, MIKMedoids)
}
```

### Ejemplo con optimización de hiperparámetros (Optuna)

```json
{
  "dataset": "musk1",
  "algoritmo": "midbscan",
  "optimizar_optuna": true,
  "optuna_trials": 30,
  "metrica_de_rendimiento_a_optimizar": "F1-Score"
}
```

---

## Estructura del proyecto

```
MIClustering/
├── src/
│   └── miclustering/
│       ├── data/
│       │   ├── midata.py          # Clase principal del dataset
│       │   ├── bag.py             # Bolsa (conjunto de instancias)
│       │   ├── instance.py        # Instancia individual
│       │   ├── attribute.py       # Esquema de atributos
│       │   └── arff_reader.py     # Lector de archivos ARFF
│       ├── distances/
│       │   ├── hausdorff.py       # Hausdorff (max, min, avg)
│       │   ├── probability_distribution.py  # EMD, Mahalanobis, Cauchy-Schwarz
│       │   ├── distance_matrix.py # Cálculo matricial de distancias
│       │   └── matrix_cache.py    # Caché persistente en disco
│       ├── models/
│       │   ├── midbscan.py        # DBSCAN para MIL
│       │   ├── mikmeans.py        # K-Means para MIL
│       │   ├── mikmedoids.py      # K-Medoids (PAM) para MIL
│       │   └── miknn.py           # k-NN para MIL
│       ├── preprocessing/
│       │   └── scaler.py          # MinMaxScaler, StandardScaler
│       └── evaluation/
│           ├── bcm.py             # Métricas de clasificación binaria
│           ├── cvi.py             # CVIs internos (SED, DD, Hc, VRC, I)
│           └── scoring.py         # Score combinado para búsqueda de hiperparámetros
├── datasets/ # Archivos ARFF (no incluidos en el repo) para tests
├── docs/                      
├── tests/
├── pyproject.toml
└── requirements.txt
```

---

## Modelos

Todos los modelos siguen la interfaz de scikit-learn (`fit`, `predict`, `fit_predict`) y operan sobre objetos `MIData`.

### MIDBSCAN

Algoritmo DBSCAN adaptado a MIL. Detecta clusters de densidad arbitraria y marca ruido sin necesidad de especificar el número de clusters.

```python
from miclustering import MIDBSCAN

model = MIDBSCAN(
    epsilon=0.5,        # Radio de vecindad
    min_pts=2,          # Mínimo de bolsas para ser punto núcleo
    metric="hausdorff"  # Métrica de distancia entre bolsas
)
model.fit(train_data)
labels = model.labels          # Dict[bag_id, cluster_id]; ruido → -1
stats  = model.get_statistics()
```

### MIKMeans

K-Means adaptado a MIL. Los centroides son bolsas sintéticas construidas como la media de todas las instancias de las bolsas asignadas al cluster.

```python
from miclustering import MIKMeans

model = MIKMeans(k=2, metric="hausdorff_avg", max_iters=100, random_state=42)
model.fit(train_data)
predictions = model.predict(test_data)
```

### MIKMedoids

K-Medoids (algoritmo PAM) adaptado a MIL. A diferencia de K-Means, los medoides son bolsas reales del dataset, lo que lo hace más robusto frente a outliers.

```python
from miclustering import MIKMedoids

model = MIKMedoids(k=2, metric="hausdorff_min", random_state=42)
model.fit(train_data)
print(model.medoids)   # Lista de bolsas que actúan como medoides
```

### MIKnn

Clasificador k-Nearest Neighbors para MIL. Lazy learning: almacena el conjunto de entrenamiento y clasifica por mayoría ponderada.

```python
from miclustering import MIKnn

model = MIKnn(k=3, metric="hausdorff")
model.fit(train_data)

predictions = model.predict(test_data)
probabilities = model.predict_proba(test_data)   # {bag_id: {0: p0, 1: p1}}
neighbors = model.get_neighbors(test_data.bags[0])  # (bag_id, label, dist)
```

---

## Métricas de distancia

| Nombre | Descripción |
|---|---|
| `hausdorff` | Hausdorff máxima (simétrica) — robusta ante outliers |
| `hausdorff_min` | Mínimo absoluto entre instancias — sensible a instancias cercanas |
| `hausdorff_avg` | Hausdorff promedio normalizada por `\|A\| + \|B\|` |
| `earth_movers` | Earth Mover's Distance — transporte óptimo entre distribuciones |
| `mahalanobis` | Mahalanobis entre distribuciones gaussianas de las bolsas |
| `cauchy_schwarz` | Similitud coseno sobre los centroides de bolsa |

Todas las métricas aceptan objetos `Bag` y devuelven un `float`. Se pueden pasar a cualquier modelo mediante el parámetro `metric`.

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

Las matrices de distancias pueden guardarse y reutilizarse automáticamente entre ejecuciones para evitar recalcularlas:

```python
from miclustering.distances.matrix_cache import global_persistent_cache

matrix = global_persistent_cache.get(
    dataset_name="musk1",
    split="train",
    scaler_name="minmax",
    metric_name="hausdorff_avg",
    bags=train_data.bags,
    metric_func=hausdorff_distance_avg,
)
```

La caché guarda los archivos en `.miclustering_cache/distance_matrices/`. La ubicación puede sobreescribirse con la variable de entorno `MICLUSTERING_CACHE_DIR`.

---

## Preprocesado

### MinMaxScaler

Escala todos los atributos numéricos al rango `[0, 1]` (configurable).

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

Ambos scalers solo actúan sobre atributos de tipo `real` o `integer`. Los atributos nominales y de cadena se conservan sin cambios.

---

## Evaluación

### Métricas de clasificación binaria (BCM)

Compara las asignaciones de cluster contra las etiquetas reales del dataset usando el algoritmo Húngaro para el mapeo óptimo cluster → clase.

```python
from miclustering.evaluation.bcm import MILEvaluator

results = MILEvaluator.evaluate(
    dataset=test_data,
    model_labels=predictions,   # Dict[bag_id, cluster_id]
    title="Experimento 1"
)
# results → {"Precision": 0.84, "Recall": 0.76, "F1-Score": 0.80, "Specificity": 0.87}
```

### Índices de validación interna (CVI)

Evalúan la calidad del clustering sin usar etiquetas, útiles para comparar configuraciones de hiperparámetros.

| Índice | Tipo | Criterio |
|---|---|---|
| `SED` | Compactibilidad | ↓ menor es mejor |
| `DD` | Compactibilidad | ↓ menor es mejor |
| `Hc` | Compactibilidad | ↓ menor es mejor |
| `VRC` | Compact. + Separación | ↑ mayor es mejor |
| `I` (PBM) | Compact. + Separación | ↑ mayor es mejor |

```python
from miclustering.evaluation.cvi import InternalCVIEvaluator

evaluator = InternalCVIEvaluator()
results = evaluator.evaluate(
    dist_matrix=distance_matrix,
    labels=model.labels,
    bag_ids=[bag.bag_id for bag in train_data.bags],
    dataset=train_scaled,   # necesario para SED, DD, VRC, I
    title="MIDBSCAN eps=0.5"
)
```

---

## Referencia de la API

### `MIData`

```python
MIData(bags: List[Bag], name: str)
MIData.from_arff(file_path, dataset_name=None, bag_column="bag", class_column="class")

dataset.get_bag(i)            # → Bag
dataset.get_num_bags()        # → int
dataset.split_data(percentage_train, seed)  # → (MIData, MIData)
dataset.get_labels()          # → List
dataset.get_positive_bags()   # → List[Bag]
dataset.get_negative_bags()   # → List[Bag]
```

### `Bag`

```python
Bag(bag_id, label, instances=None)

bag.get_instance(i)      # → Instance
bag.get_num_instances()  # → int
bag.add_instance(inst)
bag.as_matrix()          # → np.ndarray (n_instances × n_features)
```
