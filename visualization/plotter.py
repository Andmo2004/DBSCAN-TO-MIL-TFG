import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from typing import Literal

from models.midbscan import MIDBSCAN
from data.midata import MIData

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