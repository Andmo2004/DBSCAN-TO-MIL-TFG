import os
import sys

# Añadir el directorio raíz al path para poder importar desde 'models' y 'data'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from typing import Literal

from models.midbscan import MIDBSCAN
from data.midata import MIData

_DEFAULT_DISTRIB_DIR = "results/bags_distrib_plots"

def plot_mil_clusters(model: MIDBSCAN, dataset: MIData, 
                      method: Literal['pca', 'tsne'] = 'pca',
                      title: str = "MIL Clustering Visualization"):
    """
    Visualiza los clústeres MIL reduciendo la dimensionalidad de las bolsas.

    :param model: (MIDBSCAN) Instancia entrenada de MIDBSCAN.
    :param dataset: (MIData)El dataset que se usó (o se predijo).
    :param method: (Literal['pca', 'tsne'])'pca' (rápido, lineal) o 'tsne' (lento, no lineal, mejor separación visual).
    :param title: (str) Título del gráfico.
    """
    if not model.is_fitted:
        print("Error: El modelo no está entrenado.")
        return

    # 1. Preparar datos: Convertir cada Bolsa en un vector (Centroide)
    # ----------------------------------------------------------------
    bag_vectors = []
    labels = []
    bag_ids = []

    model_labels = model.labels # Obtenemos el diccionario {bag_id: label}

    for bag in dataset.bags:
        # Si la bolsa tiene etiqueta en el modelo, la procesamos
        if bag.bag_id in model_labels:
            # Estrategia: Promedio de todas las instancias de la bolsa
            # bag.as_matrix() devuelve (N_instancias, N_features)
            # np.mean(..., axis=0) devuelve (N_features,) -> Un solo punto promediado
            bag_centroid = np.mean(bag.as_matrix(), axis=0)
            
            bag_vectors.append(bag_centroid)
            labels.append(model_labels[bag.bag_id])
            bag_ids.append(bag.bag_id)

    X = np.array(bag_vectors)
    y = np.array(labels)

    # 2. Reducción de Dimensionalidad (166D -> 2D)
    # ----------------------------------------------------------------
    print(f"Reduciendo dimensiones con {method.upper()}...")
    
    if method == 'pca':
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(X)
    else:
        # t-SNE funciona mejor para ver agrupaciones separadas, pero es más lento
        # perplexity debe ser menor que el número de muestras
        perp = min(30, len(X) - 1)
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42)
        coords = reducer.fit_transform(X)

    # 3. Graficar con Matplotlib / Seaborn
    # ----------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    
    # Crear una paleta de colores
    # Filtramos el ruido para pintarlo diferente
    unique_labels = sorted(list(set(y)))
    has_noise = model.NOISE_LABEL in unique_labels
    
    # Generamos paleta: El ruido (-1) lo pondremos en gris/negro, el resto colores vivos
    palette = sns.color_palette("bright", len(unique_labels))
    color_map = dict(zip(unique_labels, palette))
    
    if has_noise:
        color_map[model.NOISE_LABEL] = (0.8, 0.8, 0.8) # Gris claro para ruido

    # Dibujar puntos
    # Iteramos para poder poner la leyenda correctamente
    for label in unique_labels:
        mask = (y == label)
        label_name = "Ruido" if label == model.NOISE_LABEL else f"Cluster {label}"
        
        plt.scatter(coords[mask, 0], coords[mask, 1], 
                    c=[color_map[label]], 
                    label=label_name,
                    alpha=0.7 if label == model.NOISE_LABEL else 1.0,
                    s=50 if label == model.NOISE_LABEL else 80, # Ruido más pequeño
                    edgecolor='w', linewidth=0.5)

    plt.title(f"{title} ({method.upper()})\nEPS={model.epsilon}, MinPts={model.min_pts}", fontsize=14)
    plt.xlabel(f"Componente 1 ({method.upper()})")
    plt.ylabel(f"Componente 2 ({method.upper()})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clusters")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_mil_clusters(model: MIDBSCAN, dataset: MIData,
                      method: Literal['pca', 'tsne'] = 'pca',
                      title: str = "MIL Clustering Visualization"):
    """
    Visualiza los clústeres MIL reduciendo la dimensionalidad de las bolsas.

    :param model: (MIDBSCAN) Instancia entrenada de MIDBSCAN.
    :param dataset: (MIData) El dataset que se usó (o se predijo).
    :param method: (Literal['pca', 'tsne']) 'pca' (rápido, lineal) o 'tsne' (lento, no lineal, mejor separación visual).
    :param title: (str) Título del gráfico.
    """
    if not model.is_fitted:
        print("Error: El modelo no está entrenado.")
        return

    # 1. Preparar datos: Convertir cada Bolsa en un vector (Centroide)
    bag_vectors = []
    labels = []
    bag_ids = []

    model_labels = model.labels

    for bag in dataset.bags:
        if bag.bag_id in model_labels:
            bag_centroid = np.mean(bag.as_matrix(), axis=0)
            bag_vectors.append(bag_centroid)
            labels.append(model_labels[bag.bag_id])
            bag_ids.append(bag.bag_id)

    X = np.array(bag_vectors)
    y = np.array(labels)

    # 2. Reducción de Dimensionalidad
    print(f"Reduciendo dimensiones con {method.upper()}...")

    if method == 'pca':
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(X)
    else:
        perp = min(30, len(X) - 1)
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42)
        coords = reducer.fit_transform(X)

    # 3. Graficar
    plt.figure(figsize=(12, 8))

    unique_labels = sorted(list(set(y)))
    has_noise = model.NOISE_LABEL in unique_labels

    palette = sns.color_palette("bright", len(unique_labels))
    color_map = dict(zip(unique_labels, palette))

    if has_noise:
        color_map[model.NOISE_LABEL] = (0.8, 0.8, 0.8)

    for label in unique_labels:
        mask = (y == label)
        label_name = "Ruido" if label == model.NOISE_LABEL else f"Cluster {label}"

        plt.scatter(coords[mask, 0], coords[mask, 1],
                    c=[color_map[label]],
                    label=label_name,
                    alpha=0.7 if label == model.NOISE_LABEL else 1.0,
                    s=50 if label == model.NOISE_LABEL else 80,
                    edgecolor='w', linewidth=0.5)

    plt.title(f"{title} ({method.upper()})\nEPS={model.epsilon}, MinPts={model.min_pts}", fontsize=14)
    plt.xlabel(f"Componente 1 ({method.upper()})")
    plt.ylabel(f"Componente 2 ({method.upper()})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clusters")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_bags_distribution(
    dataset: MIData,
    method: Literal['pca', 'tsne'] = 'tsne',
    title: str = None,
    output_dir: str = _DEFAULT_DISTRIB_DIR,
    filename: str = None,
    show: bool = False,
) -> str:
    """
    Visualiza la distribución real de bolsas (sin modelo) reduciendo
    dimensionalidad y coloreando por etiqueta de clase ground-truth.

    Guarda el PNG en output_dir y opcionalmente lo muestra.

    :param dataset:    MIData (preferiblemente ya escalado).
    :param method:     'pca' o 'tsne'.
    :param title:      Título del gráfico. Si None se genera automáticamente.
    :param output_dir: Directorio de salida (relativo a project_root).
    :param filename:   Nombre del archivo sin extensión. Si None se genera desde dataset.name.
    :param show:       Si True, muestra la figura interactivamente.
    :returns:          Ruta absoluta del PNG guardado.
    """
    # Construir ruta de salida relativa al project_root
    save_dir = os.path.join(project_root, output_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Centroides de bolsas
    bag_vectors = []
    labels = []

    for bag in dataset.bags:
        mat = bag.as_matrix()
        if len(mat) == 0:
            continue
        bag_vectors.append(np.mean(mat, axis=0))
        labels.append(int(float(bag.label)))

    if not bag_vectors:
        raise ValueError("No se encontraron bolsas con instancias en el dataset.")

    X = np.array(bag_vectors)
    y = np.array(labels)

    # Reducción de dimensionalidad
    print(f"[{dataset.name}] Reduciendo dimensiones con {method.upper()} ({len(X)} bolsas)...")

    if method == 'pca':
        coords = PCA(n_components=2).fit_transform(X)
    else:
        perp = min(30, len(X) - 1)
        coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    unique_classes = sorted(set(y))
    palette = {0: "steelblue", 1: "tomato"}
    # Fallback para datasets con más de 2 clases
    extra_colors = sns.color_palette("bright", max(0, len(unique_classes) - 2))
    for i, cls in enumerate(unique_classes):
        if cls not in palette:
            palette[cls] = extra_colors[i - 2]

    for cls in unique_classes:
        mask = y == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=palette[cls],
            label=f"Clase {cls}  (n={mask.sum()})",
            alpha=0.8, s=60, edgecolor='w', linewidth=0.5,
        )

    plot_title = title or f"{dataset.name} — distribución real de clases ({method.upper()})"
    ax.set_title(plot_title, fontsize=13)
    ax.set_xlabel(f"Componente 1 ({method.upper()})")
    ax.set_ylabel(f"Componente 2 ({method.upper()})")
    ax.legend(title="Etiqueta real")
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()

    # Guardar
    if filename is None:
        safe_name = dataset.name.replace(" ", "_").replace("/", "-")
        filename = f"{safe_name}_{method}"

    output_path = os.path.join(save_dir, f"{filename}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[{dataset.name}] Plot guardado → {output_path}")

    if show:
        plt.show()

    plt.close(fig)
    return os.path.abspath(output_path)



if __name__ == '__main__':
    from preprocessing.scaler import MinMaxScaler

    files = [
        "datasets/Thioredoxin.arff",
        "datasets/BirdsHammondsFlycatcher.arff",
    ]

    for file_path in files:

        dataset = MIData.from_arff(file_path)

        plot_bags_distribution(
            dataset=dataset,
            method='tsne',          
            show=False,             
        )

        scaler = MinMaxScaler()
        dataset = scaler.fit_transform(dataset)

        plot_bags_distribution(
            dataset=dataset,
            method='tsne',          
            show=False,             
        )