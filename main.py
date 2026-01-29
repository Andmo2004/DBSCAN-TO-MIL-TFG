from data.midata import MIData
from models.midbscan import MIDBSCAN
from visualization.plotter import plot_mil_clusters
from evaluation.evaluator import MILEvaluator
import logging
import sys

# ==========================================
# CONFIGURACIÓN DE LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mil_execution.log'),  # Guarda log en archivo
        logging.StreamHandler(sys.stdout)          # Muestra log en consola
    ]
)

logger = logging.getLogger(__name__)

def main(file_path = 'datasets/musk1.arff'):
    """
    Pipeline completo de Multi-Instance Learning:
    Carga -> Preprocesamiento -> Clustering -> Visualización -> Evaluación
    """
    logger.info("=" * 70)
    logger.info("INICIANDO PIPELINE MIL - DBSCAN")
    logger.info("=" * 70)
    
    # Variables para almacenar los datasets
    train_data = None
    test_data = None
    
    # ==========================================
    # 1. CARGA DE DATOS Y SPLIT
    # ==========================================
    logger.info("\nPASO 1: Carga y División de Datos")
    logger.info("-" * 30)
    
    try:
        # Cargamos el dataset completo
        # NOTA: Asegúrate de que la ruta al archivo .arff sea correcta
        full_dataset = MIData.from_arff(file_path)
        
        logger.info(f"Dataset cargado exitosamente: {full_dataset}")
        logger.info(f"Total bolsas: {len(full_dataset)}")
        
        # Estadísticas básicas
        positives = len(full_dataset.get_positive_bags())
        negatives = len(full_dataset.get_negative_bags())
        logger.info(f"Balance de clases: {positives} Positivas / {negatives} Negativas")

        # Dividimos en Train (70%) y Test (30%)
        logger.info("Dividiendo dataset (70% Train, 30% Test)...")
        train_data, test_data = full_dataset.split_data(percentage_train=70, seed=42)
        
        logger.info(f"Train Set: {len(train_data)} bolsas")
        logger.info(f"Test Set:  {len(test_data)} bolsas")

    except FileNotFoundError:
        logger.critical(f"No se encontró el archivo: {file_path}")
        return # Terminamos si no hay datos
    except Exception as e:
        logger.critical(f"Error crítico en la carga de datos: {e}", exc_info=True)
        return

    # ==========================================
    # 2. CONFIGURACIÓN Y ENTRENAMIENTO (FIT)
    # ==========================================
    logger.info("\nPASO 2: Entrenamiento del Modelo (Clustering)")
    logger.info("-" * 30)

    # Configuración de Hiperparámetros
    # Ajusta 'epsilon' según tus resultados anteriores. 
    # 900.0 funcionaba (agrupaba mucho), prueba bajar a 600.0 si quieres separar más.
    epsilon = 900.0 
    min_pts = 2
    
    try:
        logger.info(f"Configurando MIDBSCAN: Epsilon={epsilon}, MinPts={min_pts}")
        dbscan = MIDBSCAN(epsilon=epsilon, min_pts=min_pts, metric='hausdorff')
        
        logger.info("Entrenando modelo...")
        dbscan.fit(train_data)
        
        # Obtenemos estadísticas internas del modelo
        stats = dbscan.get_statistics()
        logger.info(f"Modelo entrenado. Clusters encontrados: {stats['num_clusters']}")
        logger.info(f"Puntos Núcleo: {stats['num_core_points']}")
        logger.info(f"Ruido en Train: {stats['noise_points_count']} bolsas")
        logger.debug(f"Distribución detallada: {stats['cluster_sizes']}")

        # ==========================================
        # 3. EVALUACIÓN DE ENTRENAMIENTO
        # ==========================================
        logger.info("\nPASO 3: Evaluación del Ajuste (Training)")
        logger.info("-" * 30)
        
        # Evaluamos qué tan bien se alinean los clusters con las etiquetas reales en Train
        MILEvaluator.evaluate(
            dataset=train_data, 
            model_labels=dbscan.labels, 
            title="Training Set (Cluster Quality)"
        )

        # ==========================================
        # 4. VISUALIZACIÓN
        # ==========================================
        logger.info("\nPASO 4: Visualización")
        logger.info("-" * 30)
        
        try:
            logger.info("Generando gráfico PCA...")
            plot_mil_clusters(
                model=dbscan, 
                dataset=train_data, 
                method='pca', 
                title=f"Musk1 Train (Eps={epsilon})"
            )
            logger.info("Gráfico generado exitosamente.")
        except Exception as e:
            logger.error(f"No se pudo generar la visualización: {e}")

        # ==========================================
        # 5. PREDICCIÓN Y EVALUACIÓN FINAL (TEST)
        # ==========================================
        logger.info("\nPASO 5: Predicción y Generalización (Test)")
        logger.info("-" * 30)
        
        if test_data and len(test_data) > 0:
            logger.info(f"Prediciendo etiquetas para {len(test_data)} bolsas de prueba...")
            
            test_results = dbscan.predict(test_data)
            
            # Evaluación final: Esta es la métrica más importante
            # Nos dice si el modelo aprendió patrones reales o solo memorizó
            MILEvaluator.evaluate(
                dataset=test_data, 
                model_labels=test_results, 
                title="Test Set (Generalization Performance)"
            )
        else:
            logger.warning("El set de prueba está vacío, saltando predicción.")

    except Exception as e:
        logger.error(f"Ocurrió un error durante el proceso de modelado: {e}", exc_info=True)

    logger.info("\n" + "="*70)
    logger.info("PROCESO FINALIZADO")
    logger.info("="*70)

if __name__ == "__main__":
    file_path = 'datasets/simple_dummy.arff'
    main(file_path)