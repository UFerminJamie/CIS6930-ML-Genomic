import scanpy as sc
import pandas as pd
import numpy as np
import NaiveDE
import SpatialDE
import glob
import os

import anndata

# Root directory where all your .h5ad files are stored
root_data_dir = './hest_datasets/her2st/st'
svg_folder = "./spatialde_results"

os.makedirs(svg_folder, exist_ok=True)

# Find all .h5ad files
h5ad_files = glob.glob(os.path.join(root_data_dir, "*.h5ad"))

for path in h5ad_files:
    sample_name = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(svg_folder, f"{sample_name}_svg_spatialde.csv")

    print(f"\n Processing {sample_name}...")
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()

    # Expression matrix
    X = adata.X
    counts = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names).T

    # Spatial coordinates
    spatial_loc = pd.DataFrame(adata.obsm["spatial"], columns=["x", "y"], index=adata.obs_names)
    
    # filter
    dup = (counts.index).duplicated()
    counts = counts.loc[dup == False, ]
    
    colzero = counts.sum(axis=0) != 0
    rowzero = counts.sum(axis=1) != 0
    counts = counts.loc[rowzero, colzero]

    spatial_loc = spatial_loc.loc[counts.columns, ]
    spatial_loc['total_counts'] = counts.sum(axis=0)
    
    # run spatialde
    norm_expr = NaiveDE.stabilize(counts) #columns are samples, rows are genes
    resid_expr = NaiveDE.regress_out(spatial_loc, norm_expr, 'np.log(total_counts)').T
    result = SpatialDE.run(spatial_loc.to_numpy(), resid_expr)
    result.to_csv(output_path, index=False)

    print(f"Saved SVGs for {sample_name} to {output_path}")
    
    
    output_txt = "./hest_datasets/her2st/processed_data/svg_200genes.txt"

    # Get all result files
    result_files = glob.glob(os.path.join(svg_folder, "*_svg_spatialde.csv"))

    gene_qvals = {}
    gene_sets = []

    for file in h5ad_files:
        adata = anndata.read_h5ad(file)
        present_genes = set(adata.var_names)
        gene_sets.append(present_genes)

    common_genes = set.intersection(*gene_sets)
    print(f"Common genes across all samples (before qval filtering): {len(common_genes)}")


    for file in result_files:
        df = pd.read_csv(file)
        
        # Filter to common genes only
        df = df[df["g"].isin(common_genes)]
        df = df[df["qval"] < 0.05]

        for _, row in df.iterrows():
            gene = row["g"]
            qval = row["qval"]
            if gene not in gene_qvals:
                gene_qvals[gene] = []
            gene_qvals[gene].append(qval)


    avg_qvals = {
        gene: sum(qvals) / len(qvals)
        for gene, qvals in gene_qvals.items()
        if len(qvals) > 0
    }

    avg_qval_df = pd.DataFrame.from_dict(avg_qvals, orient="index", columns=["avg_qval"])
    avg_qval_df = avg_qval_df.sort_values("avg_qval")

    # Top 200 genes
    top_200_genes = avg_qval_df.head(200).index.tolist()

    with open(output_txt, "w") as f:
        for gene in top_200_genes:
            f.write(gene + "\n")

    print(f"Top 200 SVGs (qval < 0.05) saved to: {output_txt}")





