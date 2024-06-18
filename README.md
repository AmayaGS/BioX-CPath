# KRAG

**Extracting complex disease signatures using graphs for  classification of heterogeneous Whole Slide Images**

## Overview 

KRAG is a self-attention hierarchical graph multiple instance learning pipeline, which combines **local, long-range and global topological** image features for optimal disease identification and subtyping. The pipeline merges **spatial** and feature** space k-nearest neighbours** to create a sparse graph, which undergoes successive self-attention hierarchical GNN and pooling layers, with added **positional encoding**. This enables the model to decide which information is more relevant for a given case, pathology or stain type, with no a priori knowledge of disease spatial presentation

![](C:\Users\Amaya\Documents\PhD\MUSTANGv2\min_code_krag\model_schema.png)

### Pipeline

- **Segmentation.** A automated tissue segmentation step, using adaptive thresholding to segment tissue areas on the WSIs.
- **Patching.** After segmentation, the tissue area is divided into patches at a size chosen by the user (eg. 224 x 224), which can be overlapping or non-overlapping.
- **Coordinates extraction.** For each patch, the (x,y)-coordinates are saved to a .csv file from the tissue segmentation.
- **Feature extraction.** Each image patch is passed through a CNN feature extractor and embedded into $[1 \times 1024]$ feature vectors. All feature vectors from a given patient are aggregated into a matrix. The number of rows in the matrix will vary as each patient has a variable set of WSIs, each with their own dimensions.
- **Adjacency matrix construction.** The patch coordinates are used to create a region Adjacency matrix $A_{RAG}$, where edges existing between spatially adjacent patches are 1 if they are spatially adjacent and 0 otherwise. The matrix of feature vectors is used to calculate the pairwise Euclidean distance between all patches. The top-k nearest neighbours in feature space are selected and a KNN Adjacency matrix $A_{KNN}$ is created, where the edges between the k-nearest neighbours is 1 and 0 otherwise.
- **Graph construction** The Adjacency matrices A_{RA} and $A_{KNN}$ are summed, with shared edges reset to 1, creating an Adjacency matrix $A_{KRAG}$. For each patient a directed, unweighted KNN+RA graph is initialised using the adjacency matrix $A_{KRAG}$, combining both local - RA - and global - KNN - information.
- **Random Walk positional encoding.** For each node in the graph, a random walk of fixed length k is performed, starting from a given node and considering only the landing probability of transitioning back to the node i itself at each step.
- **Hierarchical Graph classification.** The KRAG is successively passed through four Graph Attention Network layers (GAT) and SAGPooling layers. The SAGPooling readouts from each layer are concatenated and passed through three MLP layers. This concatenated vector is passed through a self-attention head and finally classified.
- **Heatmap generation.** Sagpool scores.

## Set Up

### General Requirements
- Python 3.10.10
- NVIDIA GPU with CUDA 12.1

### Conda Environment
```bash
conda create -n krag python=3.10.10 -y
conda activate krag

# OpenSlide
conda install -y -c conda-forge openslide openslide-python

# PyTorch (Geometric)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

conda install -y matplotlib
```

## Usage

### Data Preprocessing

During preprocessing, the following steps are performed: **tissue segmentation**, **patching**, **feature extraction**, **adjacency matrix construction**, and **graph construction**. Finally, **random walk positional encoding** is pre-computed on the generated graphs and stored as a pytorch geometric transform. 

#### Configuration File

The `config.yaml` file contains all the parameters needed for running KRAG. Modification of the input directory, output directory, and other parameters can be done there.

#### Data Directory Structure

The WSIs should be stored in a directory structure as shown below. The `slides` folder is the `input_directory`, which the `config` file should point to. It should contain all the WSIs for each patient, with the naming convention `patientID_staintype.tiff`. The `patient_labels.csv` file should contain the patient IDs and the target labels for the task:

```
--- Dataset_name
    patient_labels.csv
    --- slides
            --- patient1_HE.tiff
            --- patient1_CD3.tiff
            --- patient1_CD138.tiff
                .
                .
            --- patientN_HE.tiff
            --- patientN_CD138.tiff
```

Preprocessing can be run using the following command:

```bash
python main.py --preprocess
```

Alternatively, each step can be run separately:

```bash
python main.py --segmentation # tissue segmentation
python main.py --embedding # Feature extraction and graph construction
python main.py --compute_rwpe # Random walk positional encoding
```


