# Roadmap de Ejecución — TFG: DBSCAN en Aprendizaje de Múltiple Instancia (MIL)

> **Objetivo general:** Evaluar y validar la implementación de MIDBSCAN como algoritmo de clustering aplicado a problemas de clasificación MIL, comparándolo con el baseline MIKnn mediante un protocolo experimental riguroso.

---

## Phase 1 — Caracterización del Problema (EDA)

**Propósito:** Justificar la dificultad intrínseca de los datasets seleccionados antes de ejecutar ningún modelo. El TFG debe demostrar que el problema no es trivial.

### 1.1 Distribuciones de distancias intra-bolsa e inter-bolsa

Para cada dataset, calcular:

- **Distancia intra-clase (within-class):** distancias entre bolsas de la misma etiqueta (positivo–positivo y negativo–negativo).
- **Distancia inter-clase (between-class):** distancias entre bolsas de distinta etiqueta (positivo–negativo).

Utilizar tanto `hausdorff_distance` como `cauchy_schwarz_distance` para obtener dos perspectivas. El código de partida es `compute_distance_matrix()` de `distances/distance_matrix.py`.

```python
# Pseudocódigo orientativo
dist_matrix = compute_distance_matrix(train_scaled.bags, metric_func)
# Separar índices de bolsas positivas y negativas según bag.label
# Extraer submatrices intra e inter clase
```

**Métrica de solapamiento a reportar:** ratio de separabilidad:

```
sep_ratio = mean(inter_class_distances) / mean(intra_class_distances)
```

Un valor cercano a 1 indica solapamiento alto (dataset difícil). Calcularlo para ambas métricas.

### 1.2 Análisis de solapamiento en el espacio de distancias

- Calcular el **coeficiente de variación** de las distancias intra-clase para detectar heterogeneidad dentro de la misma clase.
- Para datasets con `sep_ratio < 1.2` (solapamiento severo), anotar como "alta dificultad" en la memoria.
- Datasets de referencia esperados en cada extremo:
  - **Baja dificultad (sep_ratio alto):** `musk1`, `mutagenesis3_atoms`
  - **Alta dificultad (solapamiento):** `BirdsChestnut`, `Thioredoxin`

### 1.3 Visualizaciones

#### Heatmaps de matrices de distancias

Usar `visualization/heatmap.py` → función `plot_distance_heatmap()`.

- Generar un heatmap por cada combinación `(dataset, métrica)` sobre el conjunto de train.
- Ordenar las bolsas por etiqueta real (primero todas las negativas, luego las positivas) para que los "bloques" de similitud sean visibles.
- Un heatmap con bloques diagonales claros indica separación natural → DBSCAN debería funcionar bien.

```python
# Reordenar bag_ids por etiqueta antes de llamar a plot_distance_heatmap()
sorted_bags = sorted(train_scaled.bags, key=lambda b: int(float(b.label)))
bag_ids_sorted = [b.bag_id for b in sorted_bags]
# Reordenar filas y columnas de dist_matrix según sorted order
```

#### Boxplots de instancias por bolsa

Para cada dataset, generar un boxplot con `matplotlib` mostrando la distribución del número de instancias por bolsa (`len(bag)` para cada `bag` en el dataset completo). Esto justifica la variabilidad y por qué métricas como Hausdorff son sensibles al tamaño de las bolsas.

**Estadísticos a incluir en la tabla resumen del EDA:**

| Dataset | n_bags | n_pos | n_neg | inst_min | inst_avg | inst_max | inst_std | sep_ratio_hau | sep_ratio_cs |
|---------|--------|-------|-------|----------|----------|----------|----------|---------------|--------------|
| musk1   | …      | …     | …     | …        | …        | …        | …        | …             | …            |
| …       |        |       |       |          |          |          |          |               |              |

### 1.4 Conclusión de la fase

Determinar para cada dataset cuál de las dos métricas (`hausdorff` o `cauchy_schwarz`) produce mayor `sep_ratio`. Esta elección debe coincidir o justificar los `best_distance` registrados en `main.py` y `tests/test_full_eval.py`.

---

## Phase 2 — Optimización de Hiperparámetros (Tuning con Optuna)

**Propósito:** Encontrar la configuración óptima `(scaler, metric, min_pts, eps)` para cada dataset de forma sistemática y reproducible.

### 2.1 Configuración del experimento

Usar `optimization/best_params.py` con los siguientes ajustes:

```python
run_optuna_search(n_trials=100)
```

- **Espacio de búsqueda:**
  - `scaler`: `{MinMaxScaler, StandardScaler}`
  - `metric`: `{hausdorff, hausdorff_min, hausdorff_avg, cauchy_schwarz, earth_movers, mahalanobis}`
  - `min_pts`: entero en `[2, 20]`
  - `eps_percentile`: float en `[1.0, 15.0]` (se convierte a eps absoluto internamente)
- **Conjunto de evaluación:** train (70% del dataset, `seed=42`).
- **Función objetivo:** F1 binario si `imbalance_ratio >= 0.3`, F1 macro si `< 0.3`.

> **Nota importante:** La caché `DistanceMatrixCache` en `best_params.py` evita recalcular la matriz de distancias para la misma combinación `(dataset, scaler, metric)`. Verificar que `global_cache._cache.clear()` se ejecuta entre datasets para no contaminar resultados.

### 2.2 Análisis de importancia de parámetros

Tras ejecutar el estudio de Optuna, generar el gráfico de importancia de parámetros:

```python
import optuna.visualization as vis
fig = vis.plot_param_importances(study)
fig.write_image(f"results/param_importance_{dataset_name}.png")
```

**Pregunta a responder:** ¿La elección del `scaler` o de la `metric` tiene más impacto en el F1? Según los resultados previos (`Notes.md`), la hipótesis es que **MinMaxScaler + Cauchy-Schwarz** domina en la mayoría de datasets porque Cauchy-Schwarz mide similitud de orientación y MinMax garantiza que todas las features contribuyan proporcionalmente al producto interno.

### 2.3 Tabla de mejores parámetros

Formato del CSV de salida (`results/optuna_best_params_<timestamp>.csv`) a incluir en la memoria:

| Dataset | Scaler | Metric | min_pts | eps_abs | best_score (F1-train) | Clusters | Noise% |
|---------|--------|--------|---------|---------|----------------------|----------|--------|
| musk1   | …      | …      | …       | …       | …                    | …        | …      |
| …       |        |        |         |         |                      |          |        |

### 2.4 Gráficos de convergencia de Optuna

Para los 3-4 datasets más representativos, incluir el **Optimization History Plot**:

```python
fig = vis.plot_optimization_history(study)
```

Muestra cómo el score mejora a lo largo de los trials, justificando que 100 trials son suficientes para la convergencia.

### 2.5 Conclusión de la fase

Determinar si existe una configuración "universal" o si el tuning es dataset-específico. Según `Notes.md`, la evidencia apunta a que **no existe configuración universal**: `BirdsChestnut` requiere `StandardScaler + Cauchy-Schwarz` con `min_pts=10`, mientras que `Newsgroups1` necesita `StandardScaler + Hausdorff`.

---

## Phase 3 — Evaluación de la Calidad del Clustering (CVIs Internos)

**Propósito:** Evaluar si los clústeres detectados por MIDBSCAN son geométricamente sólidos, independientemente de las etiquetas reales. Esto es fundamental para validar DBSCAN como algoritmo no supervisado.

### 3.1 CVIs a calcular

Todos implementados en `evaluation/cvi.py`:

| Índice | Tipo | Criterio | Requiere X |
|--------|------|----------|-----------|
| SED    | Compactibilidad | ↓ menor es mejor | Sí |
| DD     | Compactibilidad (normalizado) | ↓ menor es mejor | Sí |
| Hc     | Entropía de tamaños | ↓ menor es mejor | No |
| VRC    | Compactibilidad + Separación (Calinski-Harabasz) | ↑ mayor es mejor | Sí |
| I (PBM)| Compactibilidad + Separación | ↑ mayor es mejor | Sí |

Ejecutar mediante `InternalCVIEvaluator`:

```python
evaluator = InternalCVIEvaluator(cvis=[SEDIndex(), DDIndex(), HcIndex(), VRCIndex(), IIndex()])
results = evaluator.evaluate(dist_matrix, model.labels, bag_ids, dataset=train_scaled)
```

> **Advertencia sobre SED y ruido:** Como se anota en `Notes.md`, un modelo con alto ruido (40%+) puede tener SED bajo artificialmente porque las bolsas ruidosas no contribuyen al cálculo. Reportar siempre SED junto con `noise_pct`.

### 3.2 Comparativa: configuración óptima vs. configuración subóptima

Para demostrar que el tuning de Phase 2 mejoró la calidad geométrica, calcular los CVIs para:

- **Modelo A:** Mejor configuración encontrada por Optuna.
- **Modelo B:** Configuración con eps × 2 (epsilon artificialmente grande, demasiado permisivo).
- **Modelo C:** Configuración con eps × 0.5 (epsilon demasiado restrictivo, exceso de ruido).

**Tabla comparativa esperada (ejemplo con musk1):**

| Configuración | Clusters | Noise% | SED ↓ | DD ↓ | Hc ↓ | VRC ↑ | I ↑ |
|---------------|----------|--------|-------|------|------|-------|-----|
| Óptima        | 8        | 40.6%  | 33.85 | …    | …    | …     | …   |
| eps × 2       | 2        | 5%     | …     | …    | …    | …     | …   |
| eps × 0.5     | 15       | 65%    | …     | …    | …    | …     | …   |

### 3.3 Correlación CVIs vs. F1-Score

Construir una tabla cruzando los valores de VRC e I con el F1-score externo para todos los datasets. Calcular la **correlación de Spearman** entre cada CVI y el F1, ya que la relación no es necesariamente lineal.

```python
from scipy.stats import spearmanr
corr_vrc, p_vrc = spearmanr(vrc_values, f1_values)
corr_i, p_i = spearmanr(i_values, f1_values)
```

**Hipótesis:** Un VRC alto debería correlacionar positivamente con F1, pero datasets MIL pueden romper esta correlación cuando los clústeres de densidad no coinciden con los límites de clase.

### 3.4 Conclusión de la fase

Determinar si la estructura de densidad detectada por DBSCAN coincide con la estructura de clases del problema MIL. Esperado según los resultados en `Notes.md`:

- Datasets donde SÍ coinciden: `musk1`, `mutagenesis3_atoms`, `BirdsHammonds`
- Datasets donde NO coinciden (estructura de densidad ≠ clases): `BirdsChestnut`, `Thioredoxin`

---

## Phase 4 — Clasificación Final y Comparativa (Baseline con MIKnn)

**Propósito:** Responder la pregunta central del TFG: ¿es MIDBSCAN competitivo frente a un clasificador supervisado de distancias en problemas MIL?

### 4.1 Protocolo experimental

- **Partición:** 70% train / 30% test, `seed=42` (misma para ambos modelos).
- **Preprocessing:** Mismos `scaler` y `metric` para ambos modelos en cada dataset.
- **MIDBSCAN:** Usar los mejores parámetros de Phase 2. El mapeo clúster → clase usa el Algoritmo Húngaro implementado en `evaluation/bcm.py` → `MILEvaluator.hungarian_map_clusters_to_labels()`.
- **MIKnn:** Usar `models/miknn.py` con `k ∈ {1, 3, 5}` y reportar el mejor k por dataset (búsqueda simple en train).

### 4.2 Métricas a reportar

Para cada modelo y dataset, calcular sobre el conjunto de **test**:

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
- **Specificity** = TN / (TN + FP)

> Para datasets desbalanceados (`imbalance_ratio < 0.3`): reportar también **F1 Macro** = media de F1 por clase. Identificados en `Notes.md`: `BirdsChestnut`, `Thioredoxin`.

### 4.3 Tabla de resultados en test

| Dataset | F1-DBSCAN | F1-KNN (best k) | Δ F1 | Clusters | Noise% | k* |
|---------|-----------|-----------------|------|----------|--------|----|
| musk1   | 0.769     | …               | …    | …        | …      | …  |
| musk2   | 0.667     | …               | …    | …        | …      | …  |
| …       |           |                 |      |          |        |    |

> Los valores de F1-DBSCAN de referencia provienen de `Notes.md` (tabla de iteraciones).

### 4.4 Matrices de confusión (datasets representativos)

Seleccionar los 3 datasets más representativos de patrones distintos:

1. **musk1** — caso exitoso de DBSCAN (F1 ≈ 0.77)
2. **BirdsHammonds** — mejora progresiva significativa (F1: 0.000 → 0.939)
3. **Thioredoxin** — caso difícil (F1 ≈ 0.33, el más bajo)

Para cada uno, mostrar las matrices de confusión de ambos modelos en formato comparativo:

```
MIDBSCAN                          MIKnn
              Pred 0   Pred 1                  Pred 0   Pred 1
Real 0    [   TN   ]  [  FP  ]    Real 0    [   TN   ]  [  FP  ]
Real 1    [   FN   ]  [  TP  ]    Real 1    [   FN   ]  [  TP  ]
```

### 4.5 Análisis cualitativo por patrón de error

Según `Notes.md`, los datasets exhiben tres patrones de fallo que deben analizarse:

- **Patrón 1 — Todo predicho como negativo** (F1=0, Specificity=1): `BirdsChestnut`, `Thioredoxin`. Causa: desbalanceo extremo, todos los clústeres mapean a clase mayoritaria.
- **Patrón 2 — Todo predicho como positivo** (Recall=1, Specificity=0): `ImageElephant`. Causa: epsilon demasiado grande, un único clúster que mapea a la clase positiva.
- **Patrón 3 — Un solo clúster** (sin separación real): `Newsgroups1`. Causa: epsilon supera la distancia media entre bolsas.

Para MIKnn, identificar si estos mismos patrones aparecen o si el modelo supervisado los supera.

### 4.6 Conclusión de la fase

Responder si DBSCAN supera a KNN en datasets con estructuras complejas o ruido. Hipótesis basada en evidencia previa:

- DBSCAN **supera** a KNN cuando existe separación natural de densidades (datasets moleculares, `mutagenesis`).
- KNN **supera** a DBSCAN en datasets de audio/imagen donde las clases se solapan en el espacio de distancias (`BirdsChestnut`).

---

## Phase 5 — Validación Estadística

**Propósito:** Proporcionar rigor científico a las comparaciones del TFG. Un F1 mayor en la tabla no es evidencia suficiente sin una prueba de significancia estadística.

### 5.1 Test de Wilcoxon de rangos con signo

El **test de Wilcoxon** es el apropiado para comparar dos clasificadores sobre múltiples datasets porque:

- Es no paramétrico (no asume distribución normal de los errores).
- Trabaja con rangos de diferencias, siendo robusto a outliers.
- Es el estándar recomendado en benchmarks de ML (Demšar, 2006).

**Hipótesis:**
- H₀: No existe diferencia significativa entre MIDBSCAN y MIKnn (mediana de diferencias = 0).
- H₁: Existe diferencia significativa (bilateral).

```python
from scipy.stats import wilcoxon

f1_dbscan = [0.769, 0.667, 0.753, 0.208, 0.984, 0.984, 0.894, 0.816, 0.786, 0.333]
f1_knn    = [...]  # Completar con resultados de Phase 4

stat, p_value = wilcoxon(f1_dbscan, f1_knn, alternative='two-sided')
print(f"Estadístico W = {stat:.4f},  p-valor = {p_value:.4f}")
```

**Criterio de rechazo:** p < 0.05 (nivel de significancia estándar en ciencias de la computación).

### 5.2 Efecto del tamaño (effect size)

Complementar el p-valor con la **r de correlación de rango de Wilcoxon** como medida del tamaño del efecto:

```
r = Z / √N
```

donde Z es la estadística estandarizada y N es el número de datasets. Interpretación: `|r| < 0.3` efecto pequeño, `0.3–0.5` moderado, `> 0.5` grande.

### 5.3 Tabla de p-valores y significancia

| Comparación | W | p-valor | Significativo (α=0.05) | Effect size r | Interpretación |
|-------------|---|---------|------------------------|---------------|----------------|
| DBSCAN vs KNN (F1) | … | … | Sí/No | … | … |
| DBSCAN vs KNN (Accuracy) | … | … | Sí/No | … | … |
| DBSCAN vs KNN (Recall) | … | … | Sí/No | … | … |

### 5.4 Análisis por subgrupos

Repetir el test de Wilcoxon separando los datasets en dos subgrupos según la dificultad identificada en Phase 1:

- **Subgrupo A — Datasets con sep_ratio > 1.5** (separación clara): `musk1`, `musk2`, `mutagenesis3_atoms`, `mutagenesis3_chains`, `Harddrive1`.
- **Subgrupo B — Datasets con sep_ratio ≤ 1.5** (solapamiento): `BirdsChestnut`, `BirdsHammonds`, `Thioredoxin`, `Newsgroups1`, `ImageElephant`.

**Hipótesis:** DBSCAN debería mostrar ventaja significativa en el subgrupo A (clústeres de densidad bien definidos) pero no en el subgrupo B.

### 5.5 Conclusión de la fase

Redactar la conclusión estadística siguiendo el formato estándar de un paper de ML:

> *"El test de Wilcoxon de rangos con signo arroja un p-valor de X.XX (W = Y, α = 0.05), [rechazando / no rechazando] la hipótesis nula de igualdad de rendimiento entre MIDBSCAN y MIKnn. El tamaño del efecto es [pequeño / moderado / grande] (r = Z). Este resultado [es / no es] estadísticamente significativo para afirmar que la adaptación de DBSCAN al paradigma MIL ofrece una mejora real sobre el clasificador de vecinos más cercanos en el conjunto de benchmarks evaluado."*

---

## Resumen de Artefactos Generados

| Fase | Artefacto | Script origen | Destino |
|------|-----------|---------------|---------|
| 1 | Heatmaps de distancias | `visualization/heatmap.py` | `results/output_heatmaps/` |
| 1 | Boxplots de instancias por bolsa | Script ad-hoc matplotlib | `results/eda/` |
| 1 | Tabla EDA con sep_ratio | Script ad-hoc | `results/eda/eda_summary.csv` |
| 2 | CSV mejores parámetros Optuna | `optimization/best_params.py` | `results/optuna_best_params_*.csv` |
| 2 | Gráficos de convergencia Optuna | `optuna.visualization` | `results/optuna_plots/` |
| 3 | CSV CVIs internos (comparativa) | `tests/test_gridsearch_sed.py` | `results/cvi_grid_*.csv` |
| 4 | CSV resultados finales en test | `tests/test_full_eval.py` | `results/full_eval_*.csv` |
| 4 | Matrices de confusión | Script ad-hoc | `results/confusion_matrices/` |
| 5 | Tabla p-valores Wilcoxon | Script ad-hoc scipy | `results/statistical_tests/` |

---

## Dependencias entre Fases

```
Phase 1 (EDA)
    │
    ▼
Phase 2 (Optuna Tuning)  ───────────────────────────────────────────┐
    │                                                               │
    ▼                                                               │
Phase 3 (CVIs)           ← usa best_params de Phase 2               │
    │                                                               │
    ▼                                                               │
Phase 4 (Clasificación)  ← usa best_params de Phase 2, MIKNN      ◄─┘
    │
    ▼
Phase 5 (Wilcoxon)       ← usa F1-scores de Phase 4
```

> Los parámetros de Phase 2 alimentan directamente las Phases 3 y 4. Si se actualizan los best_params (nuevo run de Optuna), se deben reejecutar las phases 3 y 4 completas.

---

*Roadmap elaborado para TFG — EPSC. Versión 1.0.*