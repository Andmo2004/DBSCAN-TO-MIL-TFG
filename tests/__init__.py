"""
tests/

Suite de tests unitarios para MIClustering.

Estructura:
    data/          Tests para módulo data (Attribute, Instance, Bag, MIData)
    distances/     Tests para módulo distances (Hausdorff, probabilidad)
    evaluation/    Tests para módulo evaluation (scoring, bcm, cvi)
    models/        Tests para módulo models (clustering, classification)
    preprocessing/ Tests para módulo preprocessing (futuros)
    run/           Tests para módulo run (pipeline execution)

Fixtures compartidos:
    - Definidos en conftest.py (nivel raíz de tests/)
    - Helpers: make_schema, make_instance, make_bag, make_dataset
    - Datasets binarios: binary_train, binary_test, tiny_train, tiny_test
    - Edge cases: empty_dataset, single_bag_dataset

Principios:
    - Todos los objetos en memoria (sin I/O)
    - scope="function" por defecto (aislamiento total)
    - Fixtures se componen jerárquicamente
"""
