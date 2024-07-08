import argparse

from tissue_segmentation.main_tissue_segmentation import tissue_segmentation
from create_embeddings.embedding_main import patch_embedding
from create_rwpe.compute_rwpe_on_graph import compute_rwpe
from train_krag_model.main_krag import train_krag
from create_heatmaps.main_krag_heatmap import heatmap_generation

parser = argparse.ArgumentParser(description="Input arguments for applying KRAG to Whole Slide Images")

# Input arguments for tissue segmentation and patching of Whole Slide Images
parser.add_argument('--input_directory', type=str, default= r"C:\Users\Amaya\Documents\PhD\Data\Test_data_KRAG\TRACTISS_H&E", help='Input data directory')
parser.add_argument('--directory', type=str, default= r"C:\Users\Amaya\Documents\PhD\Data\Test_data_KRAG", help='Location of patient label df and extracted patches df. Embeddings and graphs dictionaries will be kept here')
parser.add_argument("--dataset_name", type=str, default="RA", choices=['RA', 'NSCLC', 'CAMELYON16', 'CAMELYON17', 'Sjogren'], help="Dataset name")
parser.add_argument('--patch_size', type=int, default=224, help='Patch size (default: 224)')
parser.add_argument('--overlap', type=int, default=0, help='Overlap (default: 0)')
parser.add_argument('--coverage', type=float, default=0.5, help='Coverage (default: 0.3)')
parser.add_argument('--slide_level', type=int, default=2, help='Slide level (default: 2)')
parser.add_argument('--mask_level', type=int, default=2, help='Slide level (default: 3)')
parser.add_argument('--unet', action='store_true', help='Calling this parameter will result in using UNet segmentation, rather than adaptive binary thresholding')
parser.add_argument('--unet_weights', type=str, default= r"C:\Users\Amaya\Documents\PhD\Data\UNet_512_1.pth.tar", help='Path to model checkpoints')
parser.add_argument('--patch_batch_size', type=int, default=10, help='Batch size (default: 10)')
parser.add_argument('--patient_ID_parsing', type=str, default='img.split("_")[0]', help='String parsing to obtain patient ID from image filename')
parser.add_argument('--stain_parsing', type=str, default='img.split("_")[1]', help='String parsing to obtain stain type from image filename')
parser.add_argument('--multistain', action= 'store_true', default=False, help='Whether the dataset contains multiple types of staining. Will generate extracted_patches.csv with stain type info.')
parser.add_argument("--seed", type=int, default=42, help="Random seed")

#Feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag].
parser.add_argument("--label", type=str, default='label', help="Name of the target label in the metadata file")
parser.add_argument("--patient_id", type=str, default='Patient_ID', help="Name of column containing the patient ID")
parser.add_argument("--K", type=int, default=7, help="Number of nearest neighbours in k-NNG created from WSI embeddings")
parser.add_argument("--embedding_vector_size", type=int, default=1000, help="Embedding vector size")
parser.add_argument("--stratified_splits", type=int, default=5, help="Number of random stratified splits")
parser.add_argument("--embedding_net", type=str, default="convnext", choices=['resnet18', 'ssl_resnet18', 'vgg16', 'convnext', 'resnet50'], help="feature extraction network used")
parser.add_argument("--embedding_weights", type=str, default=r"C:\Users\Amaya\Documents\PhD\MUSTANGv2\min_code_krag\tenpercent_resnet18.pt", help="Path to embedding weights")
parser.add_argument("--train_fraction", type=float, default=0.8, help="Train fraction")
parser.add_argument("--graph_mode", type=str, default="krag", choices=['knn', 'rag', 'krag'], help="Change type of graph used for training here")
parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
parser.add_argument("--slide_batch", type=int, default=1, help="Slide batch size - default 1")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
parser.add_argument('--stain_type', type=str, default='all', help='Type of stain used.')

#pre-compute Random Walk positional encoding on the graph
parser.add_argument("--encoding_size", type=int, default=5, help="Size Random Walk positional encoding")

#self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level"
parser.add_argument("--hidden_dim", type=int, default=512, help="Size of hidden network dimension")
parser.add_argument("--convolution", type=str, default="GAT", choices=['GAT', 'GCN', 'GIN', 'GraphSAGE'], help="Change type of graph convolution used")
parser.add_argument("--positional_encoding", default=True, help="Add Random Walk positional encoding to the graph")
parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning rate")
parser.add_argument('--attention', action='store_true', help='This parameter will result in using an attention mechanism after the graph pooling layer')
parser.add_argument("--pooling_ratio", type=float, default=0.7, help="Pooling ratio")
parser.add_argument("--heads", type=int, default=2, help="Number of GAT heads")
parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Graph batch size for training")
parser.add_argument("--scheduler", type=str, default=1, help="learning rate schedule")
parser.add_argument("--checkpoint", action="store_true", default=True, help="Enables checkpointing of GNN weights.")
parser.add_argument("--l1_norm", type=int, default=0.00001, help="L1-norm to regularise loss function")

# heatmap generation for WSI
parser.add_argument("--path_to_patches", type=str, default="/data/scratch/wpw030/KRAG/results/patches/", help="Location of patches")
parser.add_argument("--heatmap_path", type=str, default="/data/scratch/wpw030/KRAG/results/heatmaps/", help="Location of saved heatmap figs")
parser.add_argument("--checkpoint_weights", type=str, default="/data/scratch/wpw030/KRAG/", help="Location of trained model weights.")
parser.add_argument("--test_fold", type=str, default="Fold_9", help="test fold")
parser.add_argument("--test_ids", action='store_true', help="Specific IDs to create heatmap on, rather than test fold.")
parser.add_argument("--slide_name", type=str, default="/data/scratch/wpw030/KRAG/slide1", help="Location of slide which to create heatmap for.")
parser.add_argument("--per_layer", action='store_true', help="If called, will create heatmaps for each layer of the GNN.")

# General arguments to determine if running preprocessing or training.
parser.add_argument("--preprocess", action='store_true', help="Run tissue segmentation, patching of WSI, embed feature vectors, graph creation & compute RWPE.")
parser.add_argument("--segmentation", action='store_true', help="Run tissue segmentation of WSI")
parser.add_argument("--embedding", action='store_true', help="Run feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]")
parser.add_argument("--compute_rwpe", action='store_true', help="Run pre-compute of Random Walk positional encoding on the graph")
parser.add_argument("--train", action='store_true', help="Run self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level")
parser.add_argument("--heatmap", action='store_true', help="Run heatmap generation for WSI, for each layer of the GNN or all together.")

args = parser.parse_args()

def main(args):

    # Run the preprocessing steps together in one go: tissue segmentation, patching of WSI, embed feature vectors, graph creation & compute RWPE.
    if args.preprocess:
        print("Running tissue segmentation of WSIs")
        # Run tissue segmentation and patching of Whole Slide Images
        tissue_segmentation(args)
        print("Done running tissue segmentation of WSIs")

        print("Running feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]")
        # Run feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]
        patch_embedding(args)
        print("Done running feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]")

        print("Running pre-compute of Random Walk positional encoding on the graph")
        # Run pre-compute of Random Walk positional encoding on the graph
        compute_rwpe(args)
        print("Done running pre-compute of Random Walk positional encoding on the graph")

    # Run the preprocessing steps individually if needed
    if args.segmentation:
        # Run tissue segmentation of WSI
        print("Running tissue segmentation of WSIs")
        tissue_segmentation(args)
        print("Done running tissue segmentation of WSIs")

    if args.embedding:
        print("Running feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]")
        # Run feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]
        patch_embedding(args)
        print("Done running feature vector extraction of the WSI patches and creation of embedding & graph dictionaries [rag, knn or krag]")

    if args.compute_rwpe:
        print("Running pre-compute of Random Walk positional encoding on the graph")
        # Run pre-compute of Random Walk positional encoding on the graph
        compute_rwpe(args)
        print("Done running pre-compute of Random Walk positional encoding on the graph")

    # Run training of the self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level
    if args.train:
        print("Start training")
        # Run self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level
        train_krag(args)
        print("Done training")

    if args.heatmap:
        print("Running heatmap generation for WSI")
        # Run heatmap generation for WSI
        heatmap_generation(args)
        print("Done generating heatmaps for WSIs")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


