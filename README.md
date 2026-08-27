# MIClustering

> Librería de clustering y clasificación para **Multi-Instance Learning (MIL)** en Python con aceleración por **GPU (CUDA & Apple Silicon MPS)** y paralelización multinúcleo.

MIClustering implementa algoritmos avanzados adaptados al paradigma MIL, donde los datos se organizan en *bolsas* (bags) que contienen múltiples instancias. Incluye distancias vectorizadas entre bolsas, aceleración de operaciones tensoriales/GEMM por GPU, preprocesado, evaluación interna y externa (CVI y BCM), pipeline de experimentación configurable mediante JSON y soporte completo para la API de **Scikit-Learn**.

---

## 📑 Tabla de contenidos

- [Instalación](#-instalación)
- [Aceleración por GPU y Paralelismo](#-aceleración-por-gpu-y-paralelismo)
- [Inicio rápido](#-inicio-rápido)
- [Ejecución desde JSON (`run_json`)](#-ejecución-desde-json-run_json)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Modelos](#-modelos)
  - [MIDBSCAN](#midbscan)
  - [COSMIC (Nuevo)](#cosmic)
  - [MIKMeans](#mikmeans)
  - [MIKMedoids](#mikmedoids)
  - [MIKnn](#miknn)
- [Métricas de distancia](#-métricas-de-distancia)
- [Preprocesado](#-preprocesado)
- [Evaluación](#-evaluación)
- [Referencia de la API](#-referencia-de-la-api)
- [Tests](#-tests)

---

## 📦 Instalación

**Requisito previo:** Python ≥ 3.8

### Instalación estándar (CPU + Multiprocessing)

```bash
pip install -e .
```

### Instalación con soporte para GPU (PyTorch & POT)

```bash
# Con soporte de PyTorch (CUDA / Apple Silicon MPS) y Optimal Transport
pip install -e ".[gpu]"

# O con todas las dependencias (GPU + Optuna + Dev)
pip install -e ".[all]"
```

---

## ⚡ Aceleración por GPU y Paralelismo

MIClustering ofrece soporte nativo para ejecución multinúcleo y aceleración por hardware:

1. **Aceleración por GPU (CUDA / Apple Silicon MPS):**
   - **Hausdorff (Max, Min, Avg):** Cálculo de matrices de distancias cruzadas mediante `torch.cdist` aprovechando operaciones BLAS/GEMM en VRAM.
   - **Mahalanobis:** Matrices de covarianza, regularización y pseudoinversas calculadas directamente en GPU.
   - **Earth Mover's Distance (EMD):** Transporte óptimo acelerado con algoritmo Sinkhorn entropic en PyTorch o `POT`.
   - **Matriz de distancias $N \times N$:** Transferencia de tensores y paralelismo masivo en GPU con `device="auto"`, `"cuda"` o `"mps"`.
2. **Paralelización Multinúcleo en CPU (`joblib`):**
   - Cómputo del triángulo superior $(i, j)$ de la matriz de distancias distribuido sobre todos los núcleos de CPU (`n_jobs=-1`).
   - Predicciones paralelas de lotes de test en `MIDBSCAN` y `MIKnn`.
3. **Fallback transparente:**
   - Si no se detecta GPU o PyTorch no está instalado, la librería degrada automáticamente a CPU + `joblib` sin interrumpir la ejecución.

---

## 🚀 Inicio rápido

```python
from miclustering import MIDBSCAN, MIKMeans, MIKMedoids, MIKnn, COSMIC
from miclustering.data.arff_reader import ArffToMIData
from miclustering.preprocessing.scaler import MinMaxScaler
from miclustering.evaluation.bcm import MILEvaluator

# 1. Cargar dataset desde ARFF
dataset = ArffToMIData.from_arff("datasets/musk1.arff")
train_data, test_data = dataset.split_data(percentage_train=70, seed=42)

# 2. Normalizar
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled  = scaler.transform(test_data)

# 3. Entrenar con MIDBSCAN acelerado por GPU / multinúcleo
model = MIDBSCAN(epsilon=0.5, min_pts=2, metric="hausdorff_avg", n_jobs=-1, device="auto")
model.fit(train_scaled)

# 4. Acceso a propiedades estándar Scikit-Learn
print("Etiquetas de entrenamiento (array):", model.labels_)

# 5. Predecir y evaluar
predictions = model.predict(test_scaled)
MILEvaluator.evaluate(test_scaled, predictions, title="MIDBSCAN — musk1")
```

---

## ⚙️ Ejecución desde JSON (`run_json`)

Para ejecutar experimentos reproducibles, MIClustering proporciona `run_json` y `run_pipeline`, capaces de leer archivos JSON y ejecutar el pipeline completo.

### Formato del archivo de configuración JSON

```json
{
  "dataset": "musk1",
  "algoritmo": "cosmic",
  "medida_de_distancia": "hausdorff",
  "metodo_de_escalado": "MinMaxScaler",
  "semilla": 42,
  "paralelismo": -1,
  "dispositivo": "auto",
  "hiperparametros": {
    "epsilon": 1.2,
    "min_pts": 3,
    "epsilon_prime": 0.8
  },
  "optimizar_optuna": false
}
```

#### Campos disponibles y aliases soportados

| Campo | Alias comunes | Tipo | Valores posibles | Default |
|---|---|---|---|---|
| `dataset` | — | string | Nombre del dataset (ej: `"musk1"`) | **requerido** |
| `algorithm` | `algoritmo`, `model` | string | `"midbscan"`, `"cosmic"`, `"mikmeans"`, `"mikmedoids"`, `"miknn"` | `"midbscan"` |
| `distance_metric` | `medida_de_distancia`, `distance` | string | `hausdorff`, `hausdorff_avg`, `hausdorff_min`, `earth_movers`, `mahalanobis`, `cauchy_schwarz` | `"hausdorff"` |
| `scaler` | `metodo_de_escalado`, `scaling_method` | string\|null | `"MinMaxScaler"`, `"StandardScaler"`, `null` | `"MinMaxScaler"` |
| `n_jobs` | `paralelismo`, `num_workers`, `num_cores` | integer | Número de procesos (-1 para todos los núcleos) | `-1` |
| `device` | `dispositivo`, `gpu`, `aceleracion` | string | `"auto"`, `"cuda"`, `"mps"`, `"cpu"` | `"cpu"` |
| `seed` | `semilla`, `random_seed` | integer | Entero positivo | `42` |
| `hyperparams` | `hiperparametros`, `hyperparameters` | object | Parámetros específicos del modelo | `{}` |
| `use_optuna` | `optimizar_optuna`, `optimize_optuna` | boolean | `true` / `false` | `false` |
| `n_trials` | `optuna_trials` | integer | Número de ensayos de Optuna | `30` |
| `train_pct` | `porcentaje_entrenamiento` | float | 0–100 | `70.0` |

---

## 🤖 Modelos

Todos los modelos heredan de `BaseEstimator` / `ClusterMixin` o `ClassifierMixin` (Scikit-Learn) y exponen la interfaz estándar:

```python
model.fit(dataset, precomputed_matrix=None)  # Ajuste del modelo
model.predict(test_dataset)                  # Predicción inductiva sobre test
model.fit_predict(X, y=None)                 # Ajuste y retorno de etiquetas
model.labels_                                # np.ndarray con las etiquetas de entrenamiento
model.labels                                 # Dict[bag_id, label]
model.get_statistics()                       # Diccionario de diagnósticos y métricas
```

### MIDBSCAN

DBSCAN adaptado a MIL para detección de agrupaciones de forma arbitraria y detección de ruido sin requerir $k$ a priori.

```python
from miclustering import MIDBSCAN

model = MIDBSCAN(epsilon=0.5, min_pts=2, metric="hausdorff", n_jobs=-1, device="auto")
model.fit(train_data)
print("Clusters detectados:", model.cluster_count)
print("Etiquetas (ruido = -1):", model.labels_)
```

### COSMIC

Algoritmo de clustering jerárquico basado en densidad (adaptación de **OPTICS** para MIL). Genera un ordenamiento de alcanzabilidad que permite extraer múltiples particiones de clústeres para cualquier $\epsilon' \le \epsilon$ sin reentrenar.

```python
from miclustering import COSMIC

# 1. Ajuste del ordenamiento de alcanzabilidad
model = COSMIC(epsilon=1.5, min_pts=3, metric="hausdorff", n_jobs=-1)
model.fit(train_data)

# 2. Extracción de clústeres a una granularidad específica
labels_fine = model.extract_clusters(epsilon_prime=0.6)
print("Clusters con eps'=0.6:", model.cluster_count)

# 3. Acceso al perfil de alcanzabilidad (Reachability Plot)
reachability_values = model.reachability_plot
ordering_ids = model.ordering
```

### MIKMeans

K-Means para MIL. Los centroides se modelan como bolsas sintéticas de 1 sola instancia que representan el vector medio de todas las instancias del clúster, preservando la compatibilidad de tipos con las métricas de distancia MIL.

```python
from miclustering import MIKMeans

model = MIKMeans(k=3, metric="hausdorff_avg", max_iters=100, random_state=42)
model.fit(train_data)
print("Centroides calculados:", model.cluster_centers_)
```

### MIKMedoids

K-Medoids (algoritmo PAM) adaptado a MIL. Utiliza bolsas reales del dataset como medoides, aportando máxima robustez frente a instancias o bolsas anómalas.

```python
from miclustering import MIKMedoids

model = MIKMedoids(k=3, metric="hausdorff_min", random_state=42, n_jobs=-1)
model.fit(train_data)
print("Medoides reales:", model.medoids)
print("Índices de medoides:", model.medoid_indices_)
```

### MIKnn

Clasificador k-Nearest Neighbors supervisado para MIL. Realiza inferencia por votación mayoritaria entre las $k$ bolsas más cercanas con desempate por distancia acumulada.

```python
from miclustering import MIKnn

model = MIKnn(k=3, metric="hausdorff", n_jobs=-1)
model.fit(train_data)
predictions = model.predict(test_data)
probabilities = model.predict_proba(test_data)
print("Clases conocidas:", model.classes_)
```

---

## 📐 Métricas de distancia

| Nombre | Descripción | Soporte GPU |
|---|---|---|
| `hausdorff` / `hausdorff_max` | Hausdorff máxima simétrica: $\max(h(A, B), h(B, A))$ | ✅ PyTorch / MPS / CUDA |
| `hausdorff_min` | Mínimo absoluto entre instancias: $\min_{a \in A, b \in B} d(a, b)$ | ✅ PyTorch / MPS / CUDA |
| `hausdorff_avg` | Hausdorff promedio normalizada por $|A| + |B|$ | ✅ PyTorch / MPS / CUDA |
| `cauchy_schwarz` | Similitud coseno entre vectores medios de bolsa | ✅ PyTorch / MPS / CUDA |
| `mahalanobis` | Distancia de Mahalanobis con covarianza combinada $\frac{1}{2}(\Sigma_a + \Sigma_b)$ | ✅ PyTorch / MPS / CUDA |
| `earth_movers` | Earth Mover's Distance (EMD) y transporte óptimo entropic Sinkhorn | ✅ POT & PyTorch GPU |

---

## 🧼 Preprocesado

### `MinMaxScaler` y `StandardScaler`
Escaladores compatibles con Scikit-Learn que operan sobre datasets `MIData` respetando tipos nominales y atributos de identificación:

```python
from miclustering.preprocessing.scaler import MinMaxScaler, StandardScaler

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled  = scaler.transform(test_data)

# Inversión de escala exacta
original = scaler.inverse_transform(test_scaled)
```

---

## 📊 Evaluación

### Métricas de Clasificación Binaria (BCM) con Mapeo Húngaro

```python
from miclustering.evaluation.bcm import MILEvaluator

metrics = MILEvaluator.evaluate(
    dataset=test_data,
    model_labels=predictions,
    title="Evaluación MIDBSCAN"
)
# {'Precision': 0.85, 'Recall': 0.80, 'F1-Score': 0.82, 'Specificity': 0.88}
```

### Índices de Validación Interna (CVI)

```python
from miclustering.evaluation.cvi import InternalCVIEvaluator

evaluator = InternalCVIEvaluator()
cvi_results = evaluator.evaluate(
    dist_matrix=distance_matrix,
    labels=model.labels,
    bag_ids=[bag.bag_id for bag in train_scaled.bags],
    dataset=train_scaled
)
```

---

## 🧪 Tests

La suite completa cuenta con **más de 760 pruebas unitarias y de integración** construidas en memoria con validación estricta:

```bash
# Ejecutar toda la suite de pruebas
pytest

# Con reporte de cobertura
pytest --cov=miclustering --cov-report=term-missing
```
