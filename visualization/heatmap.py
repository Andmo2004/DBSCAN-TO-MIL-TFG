import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─── Directorio de salida por defecto ────────────────────────────────────────
_DEFAULT_OUTPUT_DIR = "results/heatmaps_output"


def _ensure_output_dir(output_dir: str) -> None:
    """Crea el directorio de salida si no existe."""
    os.makedirs(output_dir, exist_ok=True)


def plot_distance_heatmap(
    distance_matrix: np.ndarray,
    bag_ids: List[str],
    title: str = "Distance Matrix Heatmap",
    metric: str = "",
    eps: Optional[float] = None,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    filename: Optional[str] = None,
    show: bool = False,
    cmap: str = "YlOrRd",
) -> str:
    """
    Genera y guarda un heatmap de la matriz de distancias.

    :param distance_matrix: Matriz cuadrada (N x N) con las distancias calculadas.
    :param bag_ids:         Lista de identificadores de cada bolsa (etiquetas de ejes).
    :param title:           Título principal del heatmap.
    :param metric:          Nombre de la métrica usada (se muestra en el subtítulo).
    :param eps:             Valor de epsilon actual (se muestra en el subtítulo).
    :param output_dir:      Directorio donde se guardará la imagen.
    :param filename:        Nombre del archivo de salida (sin extensión).
                            Si es None se genera automáticamente desde el título.
    :param show:            Si True, muestra la figura interactivamente (útil en notebooks).
    :param cmap:            Colormap de matplotlib a utilizar.
    :returns:               Ruta absoluta del archivo PNG generado.
    """

    n = len(bag_ids)
    if distance_matrix.shape != (n, n):
        raise ValueError(
            f"La matriz de distancias tiene forma {distance_matrix.shape} "
            f"pero se esperaba ({n}, {n}) según los bag_ids proporcionados."
        )

    _ensure_output_dir(output_dir)

    # ── Figura ────────────────────────────────────────────────────────────
    # Escalamos el tamaño dinámicamente para que las etiquetas no se pisen
    fig_size = max(6, min(n * 0.55, 22))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # ── Imagen del heatmap ────────────────────────────────────────────────
    im = ax.imshow(distance_matrix, cmap=cmap, aspect="auto", interpolation="nearest")

    # ── Barra de color ────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Distancia", fontsize=11)

    # ── Línea de eps en la barra de color ─────────────────────────────────
    if eps is not None:
        d_min = distance_matrix.min()
        d_max = distance_matrix.max()
        if d_min <= eps <= d_max:
            cbar.ax.axhline(
                y=(eps - d_min) / (d_max - d_min),
                color="dodgerblue",
                linewidth=2,
                linestyle="--",
                label=f"ε = {eps}",
            )
            cbar.ax.text(
                1.35, (eps - d_min) / (d_max - d_min),
                f"ε={eps}",
                va="center", ha="left",
                fontsize=9, color="dodgerblue",
                transform=cbar.ax.transAxes,
            )

    # ── Anotaciones numéricas (solo si N es manejable) ────────────────────
    if n <= 30:
        d_min, d_max = distance_matrix.min(), distance_matrix.max()
        threshold = (d_max - d_min) / 2 + d_min   # umbral para texto claro/oscuro

        for i in range(n):
            for j in range(n):
                val = distance_matrix[i, j]
                color = "white" if val > threshold else "black"
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=max(4, min(8, int(80 / n))),
                    color=color,
                )

    # ── Ejes ──────────────────────────────────────────────────────────────
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    fontsize_ticks = max(5, min(10, int(120 / n)))
    ax.set_xticklabels(bag_ids, rotation=45, ha="right", fontsize=fontsize_ticks)
    ax.set_yticklabels(bag_ids, fontsize=fontsize_ticks)

    # ── Highlight: celdas dentro del radio epsilon ─────────────────────────
    if eps is not None:
        for i in range(n):
            for j in range(n):
                if i != j and distance_matrix[i, j] <= eps:
                    ax.add_patch(
                        Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False,
                            edgecolor="dodgerblue",
                            linewidth=0.8,
                            alpha=0.7,
                        )
                    )

    # ── Títulos ───────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

    subtitle_parts = []
    if metric:
        subtitle_parts.append(f"Métrica: {metric}")
    if eps is not None:
        subtitle_parts.append(f"ε = {eps}")
    subtitle_parts.append(f"{n} bolsas")

    ax.set_xlabel(" | ".join(subtitle_parts), fontsize=10, labelpad=10)

    # ── Guardado ──────────────────────────────────────────────────────────
    if filename is None:
        safe_title = title.replace(" ", "_").replace("/", "-").replace("|", "-")
        filename = f"heatmap_{safe_title}"

    output_path = os.path.join(output_dir, f"{filename}.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Heatmap guardado en: {output_path}")

    if show:
        plt.show()

    plt.close(fig)
    return os.path.abspath(output_path)