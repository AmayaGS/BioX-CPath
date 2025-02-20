import torch
import argparse
from models.MUSTANG_model import MUSTANG_Classifier
from model_stats_utils import print_model_stats
from torch_geometric.data import Data, Batch


def create_sample_data(num_nodes, num_edges, feature_dim, num_classes, walk_length):
    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, num_classes, (1,))

    # Create random walk positional encoding
    random_walk_pe = torch.randn(num_nodes, walk_length)

    data = Data(x=x, edge_index=edge_index, y=y, random_walk_pe=random_walk_pe)

    # Create a batch with a single graph
    batch = Batch.from_data_list([data])
    return batch


def main(args):
    # Create a sample graph
    sample_data = create_sample_data(args.num_nodes,
                                     args.num_edges,
                                     args.embedding_vector_size,
                                     args.n_classes,
                                     args.encoding_size)

    # Initialize the model
    model = MUSTANG_Classifier(
        in_features=args.embedding_vector_size,
        hidden_dim=args.hidden_dim,
        num_classes=args.n_classes,
        heads=args.heads,
        pooling_ratio=args.pooling_ratio,
        walk_length=args.encoding_size,
        conv_type=args.convolution
    )

    # Print model statistics
    num_params, gflops = print_model_stats(model, sample_data)
    print(f"Number of trainable parameters: {num_params}")
    print(f"Estimated GFLOPs: {gflops:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate KRAG model statistics")
    parser.add_argument("--embedding_vector_size", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--pooling_ratio", type=float, default=0.7)
    parser.add_argument("--encoding_size", type=int, default=24)
    parser.add_argument("--convolution", type=str, default="GAT")
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--num_edges", type=int, default=500)

    args = parser.parse_args()
    main(args)