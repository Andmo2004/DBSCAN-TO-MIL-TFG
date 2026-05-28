# Test Architecture & Fixtures Guide

## 📁 Estructura Actual

```
tests/
├── __init__.py                  # Suite documentation
├── conftest.py                  # ✅ Global fixtures + model fixtures
├── data/
│   ├── __init__.py
│   ├── test_attribute_instance.py
│   ├── test_bag.py
│   ├── test_midata.py
│   └── test_utils_and_arff.py
├── distances/
│   ├── __init__.py
│   ├── test_hausdorff.py
│   └── test_probability_distribution.py
├── evaluation/
│   ├── __init__.py
│   └── test_evaluation.py
├── models/
│   ├── __init__.py
│   ├── conftest.py              # ✅ Local helpers (_schema, _make_binary_dataset)
│   ├── test_midbscan.py         # ✅ Updated imports
│   ├── test_mikmeans.py         # ✅ Updated imports
│   ├── test_mikmedoids.py       # ✅ Updated imports
│   ├── test_miknn.py            # ✅ Updated imports
│   └── test_models_common.py    # ✅ Updated imports
├── preprocessing/
│   └── __init__.py              # Ready for future tests
├── run/
│   ├── __init__.py
│   └── test_run.py
└── ARCHITECTURE.md              # This file
```

## 🎯 Fixtures Disponibles

### Global (tests/conftest.py)

**Data Construction Helpers:**
```python
make_schema(n_features=3) -> list[Attribute]
make_instance(values, schema=None) -> Instance
make_bag(...) -> Bag
make_dataset(n_bags=6, n_instances=4, n_features=3, seed=0) -> MIData
```

**Fixed Fixtures:**
```python
@pytest.fixture()
def real_attribute() -> Attribute          # "feature_0" (real)
def nominal_attribute() -> Attribute       # "class" (nominal: neg, pos)
def schema_3f() -> list[Attribute]         # 3 real features
def basic_instance(schema_3f) -> Instance  # [1.0, 2.0, 3.0]
def zero_instance(schema_3f) -> Instance   # [0.0, 0.0, 0.0]
def basic_bag(schema_3f) -> Bag            # 3 instances
def empty_bag() -> Bag                     # 0 instances
def singleton_bag(schema_3f) -> Bag        # 1 instance
def small_dataset() -> MIData              # 6 bags (seed=42)
def binary_dataset_10() -> MIData          # 10 bags (seed=7)
```

**Model-Specific Fixtures (NEW):**
```python
@pytest.fixture()
def binary_train() -> MIData               # 10 pos + 10 neg (seed=42)
def binary_test() -> MIData                # 5 pos + 5 neg (seed=99)
def tiny_train() -> MIData                 # 3 pos + 3 neg (seed=7)
def tiny_test() -> MIData                  # 2 pos + 2 neg (seed=8)
def empty_dataset() -> MIData              # 0 bags
def single_bag_dataset() -> MIData         # 1 bag
```

### Local (tests/models/conftest.py)

**Private Helpers:**
```python
_schema(n=4) -> list[Attribute]
_make_bag_custom(bag_id, label, matrix, schema=None) -> Bag
_make_binary_dataset(n_pos=10, n_neg=10, n_inst=5, n_feat=4, seed=0) -> MIData
```

**Clusters Layout:**
- Positivos: cluster en [2, 3] (rng.rand() + 2.0)
- Negativos: cluster en [0, 1] (rng.rand())

## ✅ Changes Made

| Change | File | Status |
|--------|------|--------|
| Consolidate fixtures | `conftest.py` | ✅ Added binary_train, binary_test, etc. |
| Add __init__.py | All subdirs | ✅ Created in data/, distances/, evaluation/, etc. |
| Create local conftest | `tests/models/conftest.py` | ✅ Created with local helpers |
| Remove conftest_models.py | N/A | ✅ Superseded by conftest.py |
| Update imports | 5 test files | ✅ `from conftest_models` → `from conftest` |

## 🎓 Usage Examples

### Test Models
```python
def test_mikmeans_converges(binary_train):
    """Uses fixture from root conftest."""
    model = MIKMeans(k=2)
    result = model.fit(binary_train)
    assert len(result) == 2

def test_custom_dataset(self):
    """Uses helper from local conftest."""
    from tests.models.conftest import _make_binary_dataset
    ds = _make_binary_dataset(n_pos=5, n_neg=5, seed=123)
    assert len(ds.bags) == 10
```

### Test Data
```python
def test_bag_creation(basic_bag):
    """Uses root conftest fixture."""
    assert len(basic_bag.instances) == 3
    assert basic_bag.bag_id == "bag_A"
```

### Test Evaluation
```python
def test_scoring(small_dataset):
    """Uses root conftest fixture."""
    score = score_labels(small_dataset, {...})
    assert 0.0 <= score <= 1.0
```

## 🚀 Future Improvements

### 1. Desglose test_evaluation.py (cuando crezca)
```
evaluation/
├── conftest.py
├── test_scoring.py
├── test_bcm.py
├── test_cvi.py
└── test_evaluator.py
```

### 2. Parametrización
```python
@pytest.fixture(params=[2, 4, 8])
def n_features(request):
    return request.param

def test_model(binary_train, n_features):
    """Runs 3 times with different dimensionalities."""
    ...
```

### 3. Pytest Markers
```python
@pytest.mark.unit
@pytest.mark.models
@pytest.mark.slow  # Slow tests
def test_something():
    ...

# Run: pytest -m "models and not slow"
```

### 4. Factory Pattern
```python
def make_custom_dataset(
    n_pos: int, 
    n_neg: int, 
    pos_center: float = 2.0,
    seed: int = 0
) -> MIData:
    """Flexible dataset factory instead of fixed fixtures."""
    ...
```

### 5. Coverage by Module
```bash
pytest --cov=miclustering.data tests/data/
pytest --cov=miclustering.models tests/models/
pytest --cov=miclustering.evaluation tests/evaluation/
```

## 📋 Principles

| Principle | Implementation |
|-----------|-----------------|
| **Isolation** | `scope="function"` (fresh objects each test) |
| **No I/O** | All objects in memory (no ARFF files) |
| **Reproducibility** | Fixed seeds (42, 7, 8, 99, 0) |
| **Composition** | Fixtures build on smaller ones |
| **Single Source** | No duplicate definitions |
| **Clarity** | Explicit documentation in docstrings |

## ✨ Validation Checklist

- ✅ All `__init__.py` files exist
- ✅ `conftest.py` (root) has all fixtures
- ✅ `conftest.py` (models/) has local helpers
- ✅ All imports updated (conftest_models → conftest)
- ✅ No syntax errors (pytest ready)
- ✅ Fixtures properly scoped and documented
- ✅ Seeds fixed for reproducibility

## 🔗 Related Files

- Root conftest: [conftest.py](./conftest.py)
- Models conftest: [models/conftest.py](./models/conftest.py)
- Data tests: [data/](./data/)
- Model tests: [models/](./models/)
