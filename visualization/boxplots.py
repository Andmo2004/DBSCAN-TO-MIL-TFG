import os
import matplotlib.pyplot as plt
import seaborn as sns
from data.midata import MIData
import logging

logger = logging.getLogger(__name__)

def plot_instances_per_bag_boxplot(dataset: MIData, dataset_name: str, output_dir: str = "results/eda"):
    """
    Genera y guarda un boxplot con la distribución del número de instancias por bolsa.
    
    :param dataset: (MIData) El dataset completo.
    :param dataset_name: (str) Nombre del dataset para el título y el archivo.
    :param output_dir: (str) Directorio donde se guardará el gráfico.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    inst_counts = [len(bag.instances) for bag in dataset.bags]
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=inst_counts, color="skyblue")
    
    # Añadir un poco de jitter o swarmplot para ver mejor la distribución si no hay muchísimas bolsas
    if len(inst_counts) <= 1000:
        sns.stripplot(y=inst_counts, color="darkblue", size=3, alpha=0.5, jitter=True)
        
    plt.title(f"Distribución de Instancias por Bolsa - {dataset_name}", fontsize=14, fontweight="bold")
    plt.ylabel("Número de Instancias", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')
    
    output_path = os.path.join(output_dir, f"boxplot_instances_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Boxplot guardado en: {output_path}")
    return output_path
