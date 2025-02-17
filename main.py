import os
import argparse

from mains.main_tissue_segmentation import tissue_segmentation
from mains.main_embedding import patch_embedding
from mains.main_rwpe import compute_rwpe
from mains.main_train_test import train_model, test_model
from mains.main_visualisation import visualise_results
from utils.embedding_utils import seed_everything
from utils.setup_utils import setup_results_and_logging, parse_dict
from utils.model_utils import create_cross_validation_splits

parser = argparse.ArgumentParser(description="Input arguments for applying KRAG to Whole Slide Images")

# Input arguments for tissue segmentation and patching of Whole Slide Images
parser.add_argument('--input_directory', type=str, default= r"C:/Users/Amaya/Documents/PhD/Data/Test_Data_RA/R4RA_slides", help='Input data directory')
parser.add_argument('--directory', type=str, default= r"C:/Users/Amaya/Documents/PhD/Data/Test_data_RA", help='Location of patient label df and extracted patches df. Embeddings and graphs dictionaries will be kept here')
parser.add_argument('--dataset_name', type=str, default='RA', choices=['RA', 'NSCLC', 'CAMELYON16', 'CAMELYON17', 'Sjogren'], help="Dataset name")
parser.add_argument('--patch_size', type=int, default=224, help='Patch size (default: 224)')
parser.add_argument('--overlap', type=int, default=0, help='Overlap (default: 0)')
parser.add_argument('--coverage', type=float, default=0.4, help='Coverage (default: 0.3)')
parser.add_argument('--slide_level', type=int, default=2, help='Slide level (default: 1)')
parser.add_argument('--mask_level', type=int, default=2, help='Slide level (default: 2)')
parser.add_argument('--unet', action='store_true', help='WIP, do not use yet - Calling this parameter will result in using UNet segmentation, rather than adaptive binary thresholding')
parser.add_argument('--unet_weights', type=str, default= r"C:/Users/Amaya/Documents/PhD/Data/UNet_512_1.pth.tar", help='Path to model checkpoints')
parser.add_argument('--patch_batch_size', type=int, default=10, help='Batch size (default: 10)')
parser.add_argument('--patient_ID_parsing', type=str, default='img.split("_")[0]', help='String parsing to obtain patient ID from image filename')
parser.add_argument('--stain_parsing', type=str, default='img.split("_")[1]', help='String parsing to obtain stain type from image filename')
parser.add_argument('--seed', type=int, default=42, help="Random seed")

#Feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag].
parser.add_argument("--label", type=str, default='label', help="Name of the target label in the metadata file")
parser.add_argument("--label_dict", type=parse_dict, default="{'0': 'Pauci-Immune', '1': 'Lymphoid/Myeloid'}", help="Dictionary mapping int labels to string labels")
parser.add_argument("--patient_id", type=str, default='Patient_ID', help="Name of column containing the patient ID")
parser.add_argument("--K", type=int, default=7, help="Number of nearest neighbours in k-NNG created from WSI embeddings")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the GNN")
parser.add_argument("--embedding_vector_size", type=int, default=1024, help="Embedding vector size")
parser.add_argument("--stratified_splits", type=int, default=5, help="Number of random stratified splits")
parser.add_argument("--embedding_net", type=str, default="UNI", choices=['resnet18', 'ssl_resnet18', 'vgg16', 'convnext', 'resnet50', 'UNI', 'GigaPath', 'Phikon', 'BiOptimus'], help="feature extraction network used")
parser.add_argument("--embedding_weights", type=str, default=r"C:/Users/Amaya/Documents/PhD/Data/WSI_foundation/", help="Path to embedding weights")
parser.add_argument("--train_fraction", type=float, default=0.8, help="Train fraction")
parser.add_argument("--val_fraction", type=float, default=0.20, help="Validation fraction")
parser.add_argument("--graph_mode", type=str, default="krag", choices=['knn', 'rag', 'krag'], help="Change type of graph used for training here")
parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
parser.add_argument("--slide_batch", type=int, default=1, help="Slide batch size - default 1")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
parser.add_argument('--stain_type', type=str, default='all', help='Type of stain used.')

# Add arguments for stain_types and edge_types
parser.add_argument('--stain_types', type=parse_dict, default="{'NA': 0, 'H&E': 1, 'CD68': 2, 'CD138': 3, 'CD20': 4}", help='Dictionary mapping stain types to integers')
parser.add_argument('--stain_colors', type=parse_dict, default="{'H&E': 'tab:pink', 'CD68': 'tab:brown', 'CD20': 'tab:blue', 'CD138': 'tab:orange'}", help='Dictionary mapping stain types to colors')
parser.add_argument('--edge_types', type=parse_dict, default="{'RAG': 0, 'KNN': 1, 'BOTH': 2}", help='Dictionary mapping edge types to integers')
parser.add_argument('--edge_colors', type=parse_dict, default="{'RAG': 'red', 'KNN': 'dodgerblue', 'BOTH': 'blueviolet'}", help='Dictionary mapping edge types to integers')

#pre-compute Random Walk positional encoding on the graph
parser.add_argument("--encoding_size", type=int, default=20, help="Size Random Walk positional encoding")

#self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level"
parser.add_argument("--hidden_dim", type=int, default=512, help="Size of hidden network dimension")
parser.add_argument("--convolution", type=str, default="GAT", choices=['GAT', 'GCN', 'GIN', 'GraphSAGE'], help="Change type of graph convolution used")
parser.add_argument("--positional_encoding", default=True, help="Add Random Walk positional encoding to the graph")
parser.add_argument("--use_node_embedding", default=False, help="Add node embedding to the feature vectors")
parser.add_argument("--use_edge_embedding", default=False, help="Add edge embedding to the GAT layer")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument('--use_attention', action='store_true', help='This parameter will result in using an attention mechanism after the graph pooling layer')
parser.add_argument("--pooling_ratio", type=float, default=0.7, help="Pooling ratio")
parser.add_argument("--heads", type=int, default=4, help="Number of GAT heads")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Graph batch size for training")
parser.add_argument("--scheduler", type=str, default=1, help="learning rate schedule")
parser.add_argument("--checkpoint", action="store_true", default=True, help="Enables checkpointing of GNN weights.")
parser.add_argument("--L2_norm", type=int, default=0.01, help="L2-norm to regularise loss function")
parser.add_argument("--hard_test", type=bool, default=False, help="If called, will test on the hardest test set")

# visualisation of heatmaps & graph layers
parser.add_argument("--path_to_patches", type=str, default=r"C:\Users\Amaya\Documents\PhD\Data\Test_Data_RA\extracted_patches_2\patches", help="Location of patches")
parser.add_argument("--test_fold", type=int, default=0, help="Test fold to generate heatmaps for")
parser.add_argument("--test_ids", nargs="+", help="Specific test IDs to generate heatmaps for")
parser.add_argument("--specific_ids", action="store_true", help="Generate heatmaps for specific test IDs")
parser.add_argument("--per_layer", action='store_true', help="If called, will create heatmaps for each layer of the GNN.")

# benchmarking against other models
parser.add_argument("--model_name", type=str, default='KRAG', choices=['KRAG', 'MUSTANG', 'CLAM', 'DeepGraphConv', 'PatchGCN', 'TransMIL', 'GTP', 'HEAT', 'CAMIL'])

# General arguments to determine if running preprocessing, training, testing, visualisation or benchmarking.
parser.add_argument("--preprocess", action='store_true', help="Run tissue segmentation, patching of WSI, embed feature vectors, graph creation & compute RWPE.")
parser.add_argument("--segmentation", action='store_true', help="Run tissue segmentation of WSI")
parser.add_argument("--embedding", action='store_true', help="Run feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]")
parser.add_argument("--compute_rwpe", action='store_true', help="Run pre-compute of Random Walk positional encoding on the graph")
parser.add_argument("--create_splits", action='store_true', help="Create train/val/test splits")
parser.add_argument("--train", action='store_true', help="Run self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level")
parser.add_argument("--val", action='store_true', help="Run validation")
parser.add_argument("--test", action='store_true', help="Run testing")
parser.add_argument("--visualise", action='store_true', help="Run heatmap & graph visualisation for WSI, for each layer of the GNN or all together.")
parser.add_argument("--benchmark", action='store_true', help="Run benchmarking against other models")

args = parser.parse_args()

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

    if args.benchmark:
        results_dir, benchmark_logger = setup_results_and_logging(args, "_benchmark")
        benchmark_logger.info("Running benchmarking against other models")
        # Run benchmarking against other models
        run_benchmark(args, results_dir, benchmark_logger)
        benchmark_logger.info("Done benchmarking against other models")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


