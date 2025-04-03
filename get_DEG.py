import os
import glob
import anndata
import numpy as np

data_dir = "./hest_datasets/her2st/st"
h5ad_files = glob.glob(os.path.join(data_dir, "*.h5ad"))

svg_folder = "./spatialde_results"
output_txt = "./hest_datasets/her2st/processed_data/deg_200genes.txt"

# Get all result files
result_files = glob.glob(os.path.join(svg_folder, "*_svg_spatialde.csv"))

gene_qvals = {}
gene_sets = []

# collect all gene sets per file
for file in h5ad_files:
    adata = anndata.read_h5ad(file)
    present_genes = set(adata.var_names)
    gene_sets.append(present_genes)
    
# intersection of genes across all samples
common_genes = set.intersection(*gene_sets)
print(f"Common genes across all samples (before qval filtering): {len(common_genes)}")

# collect qvals ONLY for common genes with qval < 0.05
file_path = "./de_markers.xlsx"
df = pd.read_excel(file_path)
df = df[df["gene"].isin(common_genes)]
top200_deg = df.sort_values(by=["p_val_adj", "avg_logFC"], ascending=[True, False]).head(200)

top200_genes = top200_deg["gene"].tolist()

# Save the top 200 genes as a text list
txt_path = "./hest_datasets/her2st/processed_data/deg_200genes.txt"
with open(txt_path, "w") as f:
    for gene in top200_genes:
        f.write(gene + "\n")
