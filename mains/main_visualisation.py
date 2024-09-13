from utils.setup_utils import seed_everything
from visualisations.krag_vis_results_generator import KRAGResultsGenerator

def visualise_results(args, results_dir, logger):
    seed_everything(args.seed)

    # Calculate edge and node attention scores for each patient, save layer data and patient graphs to disk
    vis_results_generator = KRAGResultsGenerator(args, results_dir, logger)
    vis_results_generator.generate_data()

    logger.info("Visualization completed.")