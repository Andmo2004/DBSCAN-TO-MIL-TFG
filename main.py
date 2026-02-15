def run(dataset_path, eps, min_samples):
    """
    Ejecuta el algoritmo DBSCAN y muestra los resultados.
    Args:
        dataset_path: Ruta al archivo del dataset
        eps: Parámetro epsilon para DBSCAN
        min_samples: Número mínimo de muestras
    """
    import os
    from data import arff_reader
    from models import midbscan
    
    # Leer dataset ARFF
    if not os.path.exists(dataset_path):
        print(f"El archivo {dataset_path} no existe.")
        return
    print(f"Cargando dataset: {dataset_path}")
    bags, attributes, relation = arff_reader.read_arff(dataset_path)

    # Ejecutar DBSCAN MIL
    print(f"Ejecutando DBSCAN MIL con eps={eps}, min_samples={min_samples}")
    dbscan = midbscan.MIDBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(bags)

    # Mostrar resultados
    print("Etiquetas de los bags:")
    for i, (bag, label) in enumerate(zip(bags, labels)):
        print(f"Bag {i}: Label {label}")

    # Resumen
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"Número de clusters encontrados: {n_clusters}")
    print(f"Número de bags ruido: {n_noise}")


if __name__ == "__main__":
    # Parámetros de ejemplo
    dataset_path = "datasets/simple_dummy.arff"  # Cambia por el dataset que desees
    eps = 0.5
    min_samples = 2

    run(dataset_path, eps, min_samples)