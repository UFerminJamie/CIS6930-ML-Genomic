# Inferring Spatial Gene Expression from Histology Using Contrastive Learning

This repository implements a contrastive learning framework for predicting spatial gene expression directly from H&E-stained histology images. The model learns to align histological and transcriptomic embeddings using a symmetric contrastive objective, enabling high-resolution molecular inference using only routinely acquired H&E images.


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/UFerminJamie/CIS6930-ML-Genomic.git
cd CIS6930-ML-Genomic
```

### 2. Create Environment

Create a new conda environment using the provided YAML file:

```bash
conda env create -f environment.yml
conda activate hest
```

---

## Data Preparation

### Step 1: Download and process data

Run the following script to download and prepare the dataset:

```bash
sbatch getdata.sh
```

This script:
- Downloads the kidney and HER2ST dataset from HEST-1k dataset
- Saves them under `hest_datasets/`
- Generates top 200 highly variable genes (HVGs) for each dataset

### Step 2: Generate gene subset lists

```bash
python get_SVG.py   # Get top 200 spatially variable genes (SVGs) for HER2ST
python get_DEG.py   # Get top 200 differentially expressed genes (DEGs) for HER2ST
```

---

## 🚀 Training and Inference

### Step 3: Train the contrastive model

Submit the training job:

```bash
sbatch train_CL.sh
```

### Step 4: Run inference

Submit the evaluation job:

```bash
sbatch eval.sh
```


