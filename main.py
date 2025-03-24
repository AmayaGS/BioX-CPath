import os
import argparse

from mains.main_tissue_segmentation import tissue_segmentation
from mains.main_embedding import patch_embedding
from mains.main_rwpe import compute_rwpe
from mains.main_train_test import train_model, test_model
from mains.main_visualisation import visualise_results
from utils.embedding_utils import seed_everything
from utils.setup_utils import setup_results_and_logging, parse_dict, load_config
from utils.model_utils import create_cross_validation_splits

def parse_arguments():

    # Step 1: Parse only the config file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Sjogren_config.yaml', help='Path to the config file')
    args, remaining_argv = parser.parse_known_args()

    # Load the config file
    config = load_config(args.config)

    # Step 2: Parse all arguments, using config for defaults
    parser = argparse.ArgumentParser()

    # ====== PATHS ======
    parser.add_argument('--input_directory', type=str, default=config['paths']['input_directory'], help='Input data directory')
    parser.add_argument('--directory', type=str, default=config['paths']['output_directory'], help='Location of patient label df and extracted patches df. Embeddings and graphs dictionaries will be kept here')
    parser.add_argument('--embedding_weights', type=str, default=config['paths']['embedding_weights'], help="Path to embedding weights")
    parser.add_argument('--path_to_patches', type=str, default=config['paths']['path_to_patches'], help="Location of patches")
    parser.add_argument('--unet_weights', type=str, default=config['paths']['unet_weights'], help='Path to UNet weights')

    # ====== DATASET ======
    parser.add_argument('--dataset_name', type=str, default=config['dataset']['name'], choices=['RA', 'Sjogren'], help="Dataset name")
    parser.add_argument('--patch_size', type=int, default=config['dataset']['patch_size'], help='Patch size')
    parser.add_argument('--overlap', type=int, default=config['dataset']['overlap'], help='Overlap')
    parser.add_argument('--coverage', type=float, default=config['dataset']['coverage'], help='Coverage')
    parser.add_argument('--slide_level', type=int, default=config['dataset']['slide_level'], help='Slide level')
    parser.add_argument('--mask_level', type=int, default=config['dataset']['mask_level'], help='Mask level')
    parser.add_argument('--patch_batch_size', type=int, default=config['dataset']['patch_batch_size'], help='Batch size for patching')
    parser.add_argument('--train_fraction', type=float, default=config['dataset']['train_fraction'], help="Train fraction")
    parser.add_argument('--val_fraction', type=float, default=config['dataset']['val_fraction'], help="Validation fraction")
    parser.add_argument('--stain_type', type=str, default=config['dataset']['stain_used'], help='Type of stain used.')
    parser.add_argument('--unet', action='store_true', help='Calling this parameter will result in using UNet segmentation, rather than adaptive binary thresholding') # TODO

    # ====== PARSING ======
    parser.add_argument('--patient_ID_parsing', type=str, default=config['parsing']['patient_ID'], help='String parsing to obtain patient ID from image filename')
    parser.add_argument('--stain_parsing', type=str, default=config['parsing']['stain'], help='String parsing to obtain stain type from image filename')
    parser.add_argument('--stain_types', type=eval, default=str(config['parsing']['stain_types']), help='Dictionary mapping stain types to integers')
    parser.add_argument('--stain_colors', type=eval, default=str(config['parsing']['stain_colors']), help='Dictionary mapping stain types to colors')
    parser.add_argument('--edge_types', type=eval, default=str(config['parsing']['edge_types']), help='Dictionary mapping edge types to integers')
    parser.add_argument('--edge_colors', type=eval, default=str(config['parsing']['edge_colors']), help='Dictionary mapping edge types to colors')

    # ====== LABELS ======
    parser.add_argument("--label", type=str, default=config['labels']['label'], help="Name of the target label in the metadata file")
    parser.add_argument("--label_dict", type=eval, default=str(config['labels']['label_dict']), help="Dictionary mapping int labels to string labels")
    parser.add_argument("--patient_id", type=str, default=config['labels']['patient_id'], help="Name of column containing the patient ID")
    parser.add_argument("--n_classes", type=int, default=config['labels']['n_classes'], help="Number of classes")

    # ====== GRAPH ======
    parser.add_argument("--K", type=int, default=config['graph']['K'], help="Number of nearest neighbours in k-NNG created from WSI embeddings")
    parser.add_argument("--graph_mode", type=str, default=config['graph']['graph_mode'], choices=['knn', 'rag', 'krag'], help="Change type of graph used for training here")
    parser.add_argument("--encoding_size", type=int, default=config['graph']['encoding_size'], help="Size Random Walk positional encoding")
    parser.add_argument("--positional_encoding", default=config['graph']['positional_encoding'], help="Add Random Walk positional encoding to the graph")

    # ====== GNN ======
    parser.add_argument("--num_layers", type=int, default=config['GNN']['num_layers'], help="Number of layers in the GNN")
    parser.add_argument("--convolution", type=str, default=config['GNN']['convolution'], choices=['GAT'], help="Change type of graph convolution used")
    parser.add_argument("--use_node_embedding", default=config['GNN']['use_node_embedding'], help="Add node embedding to the feature vectors")
    parser.add_argument("--use_edge_embedding", default=config['GNN']['use_edge_embedding'], help="Add edge embedding to the GAT layer")
    parser.add_argument('--use_attention', action='store_true', default=config['GNN']['use_attention'], help='Use attention mechanism after the graph pooling layer - adds instability, remove if not needed')
    parser.add_argument("--pooling_ratio", type=float, default=config['GNN']['pooling_ratio'], help="Pooling ratio")
    parser.add_argument("--heads", type=int, default=config['GNN']['attention_heads'], help="Number of GAT heads")
    parser.add_argument("--dropout", type=float, default=config['GNN']['dropout'], help="Dropout rate")

    # ====== TRAINING ======
    parser.add_argument("--hidden_dim", type=int, default=config['training']['hidden_dim'], help="Size of hidden network dimension")
    parser.add_argument("--learning_rate", type=float, default=config['training']['learning_rate'], help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=config['training']['num_epochs'], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config['training']['batch_size'], help="Graph batch size for training")
    parser.add_argument("--slide_batch", type=int, default=config['training']['slide_batch'], help="Slide batch size")
    parser.add_argument("--num_workers", type=int, default=config['training']['num_workers'], help="Number of workers for data loading")
    parser.add_argument("--scheduler", type=str, default=config['training']['scheduler'], help="Learning rate schedule")
    parser.add_argument("--checkpoint", action="store_true", default=config['training']['checkpoint'], help="Enables checkpointing of GNN weights")
    parser.add_argument("--L2_norm", type=float, default=config['training']['L2_norm'], help="L2-norm to regularise loss function")
    parser.add_argument("--hard_test", type=bool, default=False, help="If called, will test on the hardest test set")
    parser.add_argument('--seed', type=int, default=config['training']['seed'], help="Random seed")
    parser.add_argument("--stratified_splits", type=int, default=config['training']['stratified_splits'], help="Number of random stratified splits")
    parser.add_argument("--weight_type", type=str, default=config['training']['weight_type'], choices=['accuracy', 'loss', 'auc'], help="model weights type")

    # ====== VISUALIZATION ======
    parser.add_argument("--test_fold", type=int, default=config['visualization']['test_fold'], help="Test fold to generate heatmaps for")
    parser.add_argument("--test_ids", nargs="+", default=config['visualization']['test_ids'], help="Specific test IDs to generate heatmaps for")
    parser.add_argument("--specific_ids", action="store_true", default=config['visualization']['specific_ids'], help="Generate heatmaps for specific test IDs")

    # ====== MODEL ======
    parser.add_argument("--model_name", type=str, default=config['model']['name'], choices=['BioXCPath', 'MUSTANG', 'CLAM', 'DeepGraphConv', 'PatchGCN', 'TransMIL', 'GTP'])
    parser.add_argument("--embedding_net", type=str, default="UNI", choices=list(config['embedding_nets'].keys()), help="Feature extraction network used")

    # ====== EXECUTION FLAGS ======
    parser.add_argument("--preprocess", action='store_true', default=config['execution']['preprocess'], help="Run tissue segmentation, patching of WSI, embed feature vectors, graph creation & compute RWPE")
    parser.add_argument("--segmentation", action='store_true', default=config['execution']['segmentation'], help="Run tissue segmentation of WSI")
    parser.add_argument("--embedding", action='store_true', default=config['execution']['embedding'], help="Run feature vector extraction of the WSI patches and creation of embedding & graph dictionaries")
    parser.add_argument("--compute_rwpe", action='store_true', default=config['execution']['compute_rwpe'], help="Run pre-compute of Random Walk positional encoding on the graph")
    parser.add_argument("--create_splits", action='store_true', default=config['execution']['create_splits'], help="Create train/val/test splits")
    parser.add_argument("--train", action='store_true', default=config['execution']['train'], help="Run training")
    parser.add_argument("--val", action='store_true', default=config['execution']['val'], help="Run validation")
    parser.add_argument("--test", action='store_true', default=config['execution']['test'], help="Run testing")
    parser.add_argument("--visualise", action='store_true', default=config['execution']['visualise'], help="Run heatmap & graph visualisation")

    args = parser.parse_args(remaining_argv)

    # Set embedding_vector_size based on the selected embedding_net
    args.embedding_vector_size = config['embedding_nets'][args.embedding_net]['size']

    return args, config

def main(args):
    
    seed_everything(args.seed)

    # Run the preprocessing steps together in one go: tissue segmentation, patching of WSI, embed feature vectors, graph creation & compute RWPE.
    if args.preprocess:
        # Setup logging
        _, preprocess_logger = setup_results_and_logging(args, "_preprocess")

        preprocess_logger.info("Running tissue segmentation of WSIs")
        # Run tissue segmentation and patching of Whole Slide Images
        tissue_segmentation(args, preprocess_logger)
        preprocess_logger.info("Done running tissue segmentation of WSIs")

        preprocess_logger.info("Running feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]")
        # Run feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]
        patch_embedding(args, preprocess_logger)
        preprocess_logger.info("Done running feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]")

        preprocess_logger.info("Running pre-compute of Random Walk positional encoding on the graph")
        # Run pre-compute of Random Walk positional encoding on the graph
        compute_rwpe(args, preprocess_logger)
        preprocess_logger.info("Done running pre-compute of Random Walk positional encoding on the graph")

        preprocess_logger.info("Creating train/val/test splits")
        sss_dict_path = os.path.join(args.directory, f"train_test_strat_splits_{args.dataset_name}.pkl")
        if not os.path.exists(sss_dict_path):
            create_cross_validation_splits(
                args,
                patient_id=args.patient_id,
                label=args.label,
                test_size=1-args.train_fraction,
                n_splits=args.stratified_splits,
                seed=args.seed,
                dataset_name=args.dataset_name,
                directory=args.directory,
                hard_test_set=args.hard_test
            )
        preprocess_logger.info("Done creating train/val/test splits")

    # Run the preprocessing steps individually if needed
    if args.segmentation:
        # Setup logging
        _, preprocess_logger = setup_results_and_logging(args, "_preprocess")
        # Run tissue segmentation of WSI
        preprocess_logger.info("Running tissue segmentation of WSIs")
        tissue_segmentation(args, preprocess_logger)
        preprocess_logger.info("Done running tissue segmentation of WSIs")

    if args.embedding:
        # Setup logging
        _, preprocess_logger = setup_results_and_logging(args, "_preprocess")
        preprocess_logger.info("Running feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]")
        # Run feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]
        patch_embedding(args, preprocess_logger)
        preprocess_logger.info("Done running feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]")

    if args.compute_rwpe:
        # Setup logging
        _, preprocess_logger = setup_results_and_logging(args, "_preprocess")
        preprocess_logger.info("Running pre-compute of Random Walk positional encoding on the graph")
        # Run pre-compute of Random Walk positional encoding on the graph
        compute_rwpe(args, preprocess_logger)
        preprocess_logger.info("Done running pre-compute of Random Walk positional encoding on the graph")

    if args.create_splits:
        # Setup logging

        _, preprocess_logger = setup_results_and_logging(args, "_preprocess")
        preprocess_logger.info("Creating train/val/test splits")
        sss_dict_path = os.path.join(args.directory, f"train_test_strat_splits_{args.dataset_name}.pkl")
        if not os.path.exists(sss_dict_path):
            create_cross_validation_splits(
                args,
                patient_id=args.patient_id,
                label=args.label,
                test_size=1-args.train_fraction,
                n_splits=args.stratified_splits,
                seed=args.seed,
                dataset_name=args.dataset_name,
                directory=args.directory
            )
        preprocess_logger.info("Done creating train/val/test splits")

    # Run training of the self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level
    if args.train:
        results_dir, train_logger = setup_results_and_logging(args, "_training")
        train_logger.info("Start training")
        # Run self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level
        train_model(args, results_dir, train_logger)
        train_logger.info("Done training")

    if args.val:
        results_dir, test_logger = setup_results_and_logging(args, "_val")
        test_logger.info("Running validation")
        test_model(args, results_dir, test_logger)
        test_logger.info("Done validation")

    if args.test:
        results_dir, test_logger = setup_results_and_logging(args, "_testing")
        test_logger.info("Running testing")
        test_model(args, results_dir, test_logger)
        test_logger.info("Done testing")

    if args.visualise:
        results_dir, vis_logger = setup_results_and_logging(args, "_visualisation")
        vis_logger.info("Running visualisation of heatmaps & graph layers")
        # Run visualisation of heatmaps & graph layers
        visualise_results(args, results_dir, vis_logger)
        vis_logger.info("Done visualising heatmaps & graph layers")

    # if args.benchmark:
    #     results_dir, benchmark_logger = setup_results_and_logging(args, "_benchmark")
    #     benchmark_logger.info("Running benchmarking against other models")
    #     # Run benchmarking against other models
    #     run_benchmark(args, results_dir, benchmark_logger)
    #     benchmark_logger.info("Done benchmarking against other models")


if __name__ == "__main__":
    args, config = parse_arguments()
    main(args)


