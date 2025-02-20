import os
import pickle
from visualisations.bioxcpath_vis_results import VisualisationResultsGenerator
from visualisations.metrics_visualiser import MetricsVisualiser
from visualisations.stain_relationship import StainRelationship
from visualisations.heatmap_generator import HeatmapGenerator
from visualisations.graph_visualiser import GraphVisualiser


def visualise_results(args, results_dir, logger):
    """
    Main visualization function that processes all folds.

    Args:
        args: Configuration arguments
        results_dir: Base directory for results
        logger: Logger instance
    """

    # Load splits if no specific test fold is provided
    if not args.specific_ids and not args.test_fold:
        logger.info("No specific patient IDs or test fold provided. Processing all folds.")
        with open(args.directory + f"/train_test_strat_splits_{args.dataset_name}.pkl", "rb") as file:
            splits = pickle.load(file)
    else:
        splits = {args.test_fold: {'Test': args.specific_ids}}

    # Initialize metrics visualizer once
    metrics_visualiser = MetricsVisualiser()
    stain_relationship = StainRelationship(args)
    heatmap_generator = HeatmapGenerator(args, results_dir, logger)
    graph_visualiser = GraphVisualiser(args, results_dir, logger)

    # Process each fold
    for fold, fold_data in enumerate(splits.items()):
        logger.info(f"Processing fold: {fold}")

        # Setup directories for this fold
        model_name = os.path.basename(results_dir)
        vis_path = os.path.join(args.directory, "vis_data", model_name)
        output_dir = os.path.join(args.directory, "graph_visualisations", model_name)
        fold_dir = os.path.join(output_dir, f"Fold_{fold}")
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(fold_dir, exist_ok=True)

        # Generate visualization data
        vis_results_generator = VisualisationResultsGenerator(args, results_dir, logger)
        all_metrics, all_patient_data = vis_results_generator.process_fold(fold, fold_data[1]['Test']) # all the metrics are being generated here

        # Generate visualizations for this fold
        if all_metrics:
            graph_visualiser.visualise_graphs(all_patient_data, fold_dir, fold)
            heatmap_generator.generate_heatmaps(all_patient_data, fold_dir, fold)
            metrics_visualiser.plot_metrics(args, all_metrics, fold_dir)
            stain_relationship.plot_stain_relationships(all_patient_data, fold_dir, 'all')
            logger.info(f"Successfully generated visualizations for fold {fold}")

    logger.info("Visualization completed.")
