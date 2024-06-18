# KRAG

## Overview 
KNN + RAG self attention graph for histopathology whole slide imaging

- **Segmentation.** A automated segmentation step, where UNet is used to segment tissue areas on the WSIs. The user can use the trained weights provided on our GitHub repository or use their own.
- **Patching.** After segmentation, the tissue area is divided into patches at a size chosen by the user, which can be overlapping or non-overlapping.
- **Coordinates extraction.** For each patch, the (x,y)-coordinates are saved from the tissue segmentation.
- **Feature extraction.** Each image patch is passed through a CNN feature extractor and embedded into $[1 \times 1024]$ feature vectors. All feature vectors from a given patient are aggregated into a matrix. The number of rows in the matrix will vary as each patient has a variable set of WSIs, each with their own dimensions.
- **Adjency matrix construction.** The patch coordinates are used to create a region adjacency Adjacency matrix $A_{RAG}$, where edges existing between spatially adjacent patches are 1 if they are spatially adjacent and 0 otherwise. The matrix of feature vectors is used to calculate the pairwise Euclidean distance between all patches. The top-k nearest neighbours in feature space are selected and a KNN Adjacency matrix $A_{KNN}$ is created, where the edges between the k-nearest neighbours is 1 and 0 otherwise.
- **Graph construction** The Adjacency matrices A_{RA} and $A_{KNN}$ are summed, with shared edges reset to 1, creating an Adjacency matrix $A_{KRAG}$. For each patient a directed, unweighted KNN+RA graph is initialised using the adjacency matrix $A_{KRAG}$, combining both local - RA - and global - KNN - information.
- **Random Walk positional encoding.** For each node in the graph, a random walk of fixed length k is performed, starting from a given node and considering only the landing probability of transitioning back to the node i itself at each step.
- **Graph classification.** The KRAG is successively passed through four Graph Attention Network layers (GAT) and SAGPooling layers. The SAGPooling readouts from each layer are concatenated and passed through three MLP layers. This concatenated vector is passed through a self-attention head and finally classified.
- **Heatmap generation.** Sagpool scores.

## Set Up

### General Requirements
- Python 3.10.10
- NVIDIA GPU with CUDA 11.8

### Conda Environment
```bash
conda create -n krag python=3.10.10 -y
conda activate krag

# Open slide
conda install -y -c conda-forge openslide openslide-python

# PyTorch (Geometric)
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y pyg -c pyg

conda install -y matplotlib
```

