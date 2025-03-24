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

    # Load all splits first
    with open(args.directory + f"/train_test_strat_splits_{args.dataset_name}.pkl", "rb") as file:
        all_splits = pickle.load(file)

    # Case 1: No specific IDs or test fold - use all splits as is
    if not args.specific_ids and not args.test_fold:
        logger.info("No specific patient IDs or test fold provided. Processing all folds.")
        splits = all_splits
    # Case 2: Specific IDs across all folds
    elif args.specific_ids:
        # Include all folds, but filter each fold's Test set for the specific IDs
        patient_ids = args.specific_ids
        splits = {}
        for fold_key, fold_data in all_splits.items():
            splits[fold_key] = {"Test": [id for id in patient_ids if id in fold_data["Test"]]}
    # Case 3: Only specific test fold (no specific IDs)
    else:
        # Include only the specific test fold with all its IDs
        splits = {}
        fold_key = f"Fold {args.test_fold}"
        splits[fold_key] = {"Test": all_splits[fold_key]["Test"]}

    # Initialize metrics visualizer once
    metrics_visualiser = MetricsVisualiser(logger)
    stain_relationship = StainRelationship(args, logger)
    heatmap_generator = HeatmapGenerator(args, results_dir, logger)
    graph_visualiser = GraphVisualiser(args, results_dir, logger)

    # Process each fold
    for fold_key, fold_data in splits.items():
        # Extract fold number for downstream processing
        if fold_key.startswith("Fold "):
            # For cases where the key is in "Fold X" format
            fold = int(fold_key.split(" ")[1])
    # for fold, fold_data in enumerate(splits.items()):
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
        all_metrics, all_patient_data = vis_results_generator.process_fold(fold, fold_data['Test']) # all the metrics are being generated here

        # Generate visualizations for this fold
        if all_metrics:
            #metrics_visualiser.plot_metrics(args, all_metrics, fold_dir) # this does both all and multistain already
            #stain_relationship.plot_stain_relationships(all_patient_data, fold_dir, 'all')
            #stain_relationship.plot_stain_relationships(all_patient_data, fold_dir, 'multistain')
            graph_visualiser.visualise_graphs(all_patient_data, fold_dir, fold)
            #heatmap_generator.generate_heatmaps(all_patient_data, fold_dir, fold)
            logger.info(f"Successfully generated visualizations for fold {fold}")

    logger.info("Visualization completed.")
