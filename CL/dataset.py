import torch 
import anndata
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
import os
from glob import glob
import logging

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)

    return logger

def create_dir(input_args):
    print("mkdir & set up logger...")
    # mkdir for logs and checkpoints
    os.makedirs(input_args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{input_args.results_dir}/*"))
    input_args.experiment_dir = f"{input_args.results_dir}/{experiment_index:03d}"  # Create an experiment folder
    input_args.checkpoint_dir = f"{input_args.experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(input_args.checkpoint_dir, exist_ok=True)
    os.makedirs(f"{input_args.experiment_dir}/samples", exist_ok=True)      # Store sampling results
    input_args.logger = create_logger(input_args.experiment_dir)
    input_args.logger.info(f"Experiment directory created at {input_args.experiment_dir}")


def create_save_dir(input_args):
    print("mkdir & set up logger...")
    # mkdir for logs and checkpoints
    os.makedirs(f"{input_args.save_path}/embeddings", exist_ok=True)      # Store sampling results
    input_args.logger = create_logger(f"{input_args.save_path}/embeddings")
    input_args.logger.info(f"Embeddings will be saved in this directory {input_args.save_path}/embeddings")




def get_spatial_x_y_mappings():
    data_paths = ['./hest_datasets/kidney/', './hest_datasets/her2st/', 
                  './hest_datasets/mouse_brain/', './hest_datasets/PRAD/']
    all_coords = []
    
    for data_path in data_paths:
        folder_list_path = data_path + "processed_data/" + "all_slide_lst.txt"
        slidename_lst = list(np.genfromtxt(folder_list_path, dtype=str))
        for sni in range(len(slidename_lst)):
            sample_name = slidename_lst[sni]
            adata_path = data_path + "st/" + sample_name + ".h5ad"
            test_adata = anndata.read_h5ad(adata_path)
            spatial_coords = test_adata.obsm['spatial']  # shape: (N, 2)
            all_coords.append(spatial_coords)
    all_coords_np = np.concatenate(all_coords, axis=0)  # shape: (total_N, 2)
    all_coords_tensor = torch.tensor(all_coords_np, dtype=torch.int32)
    
    x_vals = all_coords_tensor[:, 0]
    y_vals = all_coords_tensor[:, 1]

    unique_x = torch.unique(x_vals)
    unique_y = torch.unique(y_vals)

    x2idx = {v.item(): i for i, v in enumerate(unique_x)}
    y2idx = {v.item(): i for i, v in enumerate(unique_y)}
    return x2idx, y2idx, unique_x, unique_y

class CustomDataset(Dataset):
    def __init__(self, x, y, coord):
        self.data = x
        self.label = y
        self.coord = coord

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.coord[:,0][idx], self.coord[:,1][idx]
    
def assemble_dataset(input_args):
    # load & assemble data
    # leave the test slide out
    slide_outs = []
    slidename_lst = list(np.genfromtxt(input_args.data_path + "processed_data/" + input_args.folder_list_filename, dtype=str))
    for slide_out in input_args.slide_out.split(","):
        slidename_lst.remove(slide_out)
        slide_outs.append(slide_out)
        input_args.logger.info(f"{slide_out} is held out for testing.")
    input_args.logger.info(f"Remaining {len(slidename_lst)} slides: {slidename_lst}")

    # load selected gene list
    selected_genes = list(np.genfromtxt(input_args.data_path + "processed_data/" + input_args.gene_list_filename, dtype=str))
    input_args.input_gene_size = len(selected_genes)
    input_args.logger.info(f"Selected genes filename: {input_args.gene_list_filename} | len: {len(selected_genes)}")

    if input_args.test:
        slidename_lst = slide_outs
    input_args.datasize = []

    # load original patches
    first_slide = True
    all_img_ebd_ori = None
    all_count_mtx_ori = None
    all_coord_ori = None
    input_args.logger.info("Loading original data...")
    for sni in range(len(slidename_lst)):
        sample_name = slidename_lst[sni]
        test_adata = anndata.read_h5ad(input_args.data_path + "st/" + sample_name + ".h5ad")
        input_args.datasize.append(test_adata.shape[0])
        test_count_mtx = pd.DataFrame(test_adata[:, selected_genes].X.toarray(), 
                                      columns=selected_genes, 
                                      index=[sample_name + "_" + str(i) for i in range(test_adata.shape[0])])
        test_coord = test_adata.obsm['spatial']

        if first_slide:
            all_count_mtx_ori = test_count_mtx
            all_coord_ori = test_coord
            img_ebd_uni   = torch.load(input_args.data_path + "processed_data/uni_ebd/"   + sample_name + "_uni.pt",   map_location="cpu")
            img_ebd_conch = torch.load(input_args.data_path + "processed_data/conch_ebd/" + sample_name + "_conch.pt", map_location="cpu")
            all_img_ebd_ori = torch.cat([img_ebd_uni, img_ebd_conch], axis=1)
            input_args.logger.info(f"{sample_name} loaded, count_mtx shape: {all_count_mtx_ori.shape}  | img ebd shape: {all_img_ebd_ori.shape}")
            first_slide = False
            continue

        img_ebd_uni   = torch.load(input_args.data_path + "processed_data/uni_ebd/"   + sample_name + "_uni.pt",   map_location="cpu")
        img_ebd_conch = torch.load(input_args.data_path + "processed_data/conch_ebd/" + sample_name + "_conch.pt", map_location="cpu")
        slide_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], axis=1)
        all_img_ebd_ori = torch.cat([all_img_ebd_ori, slide_img_ebd], axis=0)
        all_count_mtx_ori = np.concatenate((all_count_mtx_ori, test_count_mtx), axis=0)
        all_coord_ori = np.concatenate((all_coord_ori, test_coord), axis=0)
        input_args.logger.info(f"{sample_name} loaded, count_mtx shape: {all_count_mtx_ori.shape} | img ebd shape: {all_img_ebd_ori.shape} | coord shape {all_coord_ori.shape}")
    input_args.cond_size = all_img_ebd_ori.shape[1]

    # load augmented patches
    first_slide = True
    all_img_ebd_aug = None
    input_args.logger.info(f"Augmentation data loading...")
    for sni in range(len(slidename_lst)):
        sample_name = slidename_lst[sni]

        if first_slide:
            img_ebd_uni   = torch.load(input_args.data_path + "processed_data/uni_ebd_aug/"   + sample_name + "_uni_aug.pt",   map_location="cpu")
            img_ebd_conch = torch.load(input_args.data_path + "processed_data/conch_ebd_aug/" + sample_name + "_conch_aug.pt", map_location="cpu")
            all_img_ebd_aug = torch.cat([img_ebd_uni, img_ebd_conch], axis=-1)
            input_args.logger.info(f"With augmentation {sample_name} loaded, img_ebd_mtx shape: {all_img_ebd_aug.shape}, all_img_ebd shape: {all_img_ebd_aug.shape}")
            first_slide = False
            continue

        img_ebd_uni   = torch.load(input_args.data_path + "processed_data/uni_ebd_aug/"   + sample_name + "_uni_aug.pt",   map_location="cpu")
        img_ebd_conch = torch.load(input_args.data_path + "processed_data/conch_ebd_aug/" + sample_name + "_conch_aug.pt", map_location="cpu")
        slide_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], axis=-1)
        all_img_ebd_aug = torch.cat([all_img_ebd_aug, slide_img_ebd], axis=0)
        input_args.logger.info(f"With augmentation {sample_name} loaded, img_ebd_mtx shape: {slide_img_ebd.shape}, all_img_ebd shape: {all_img_ebd_aug.shape}")

    # randomly select augmented patches according to the input augmentation ratio (int)
    num_aug_ratio = input_args.num_aug_ratio
    all_count_mtx_aug = np.repeat(np.copy(all_count_mtx_ori), num_aug_ratio, axis=0)             # generate count matrix for all augmented patches
    all_coord_aug = np.repeat(np.copy(all_coord_ori), num_aug_ratio, axis=0)   
    selected_img_ebd_aug = torch.zeros((all_count_mtx_aug.shape[0], all_img_ebd_aug.shape[2]))
    for i in range(all_img_ebd_aug.shape[0]):                                                    # randomly select augmented patches
        selected_transpose_idx = np.random.choice(all_img_ebd_aug.shape[1], num_aug_ratio, replace=False)
        selected_img_ebd_aug[i*num_aug_ratio:(i+1)*num_aug_ratio, :] = all_img_ebd_aug[i, selected_transpose_idx, :]

    all_img_ebd = torch.cat([all_img_ebd_ori, selected_img_ebd_aug], axis=0)
    all_count_mtx = np.concatenate((all_count_mtx_ori, all_count_mtx_aug), axis=0)
    all_coord = np.concatenate((all_coord_ori, all_coord_aug), axis=0)
    input_args.logger.info(f"{num_aug_ratio}:1 augmentation. CONCH+UNI. final count_mtx shape: {all_count_mtx.shape} | final img_ebd shape: {all_img_ebd.shape} | final coord shape: {all_coord.shape}")

    ################################################
    all_count_mtx_df = pd.DataFrame(all_count_mtx, columns=selected_genes, index=list(range(all_count_mtx.shape[0])))

    if input_args.test:
        all_count_mtx = all_count_mtx_df
        all_coord = torch.from_numpy(all_coord)
    else:
        # remove the spot with all NAN/zero in count mtx
        all_count_mtx_all_nan_spot_index = all_count_mtx_df.index[all_count_mtx_df.isnull().all(axis=1)]
        all_count_mtx_all_zero_spot_index = all_count_mtx_df.index[all_count_mtx_df.sum(axis=1) == 0]
        input_args.logger.info(f"All NAN spot index: {all_count_mtx_all_nan_spot_index}")
        input_args.logger.info(f"All zero spot index: {all_count_mtx_all_zero_spot_index}")
        spot_idx_to_remove = list(set(all_count_mtx_all_nan_spot_index) | set(all_count_mtx_all_zero_spot_index))
        spot_idx_to_keep = list(set(all_count_mtx_df.index) - set(spot_idx_to_remove))
        all_count_mtx = all_count_mtx_df.loc[spot_idx_to_keep, :]
        all_img_ebd = all_img_ebd[spot_idx_to_keep, :]
        all_coord = torch.from_numpy(all_coord[spot_idx_to_keep, :])
    
    input_args.logger.info(f"After exclude rows with all nan/zeros: {all_count_mtx.shape}, {all_img_ebd.shape}, {all_coord.shape}")
    # only normalized by log2(+1)
    # all_count_mtx_selected_genes = all_count_mtx_selected_genes.loc[:, ~all_count_mtx_selected_genes.columns.duplicated()]
    # print
    # print(all_count_mtx.loc[:, selected_genes].shape)
    # print(np.log2(all_count_mtx.loc[:, selected_genes] + 1).copy().shape)
    all_count_mtx_selected_genes = np.log2(all_count_mtx + 1).copy()
    input_args.logger.info(f"Selected genes count matrix shape: {all_count_mtx_selected_genes.shape}")
    all_img_ebd.requires_grad_(False)
    all_coord.requires_grad_(False)
    alldataset = CustomDataset(torch.from_numpy(all_count_mtx_selected_genes.values).float(), 
                               all_img_ebd.float(), all_coord)    
    
    return alldataset, input_args


def load_data(input_args):
    alldataset, input_args = assemble_dataset(input_args)
    loader = DataLoader(alldataset, batch_size=input_args.batch_size, shuffle=True)
    return loader, input_args



# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--expr_name", type=str, default="PRAD")
#     parser.add_argument("--data_path", type=str, default="./hestk_datasets/PRAD/", help="Dataset path")
#     parser.add_argument("--results_dir", type=str, default="./PRAD_results/CL/runs/", help="Path to hold runs")
#     parser.add_argument("--slide_out", type=str, default="MEND145", help="Test slide ID. Multiple slides separated by comma.") 
#     parser.add_argument("--folder_list_filename", type=str, default="all_slide_lst.txt", help="A txt file listing file names for all training and testing slides in the dataset")
#     parser.add_argument("--gene_list_filename", type=str, default="selected_gene_list.txt", help="Selected gene list")
#     parser.add_argument("--num_aug_ratio", type=int, default=7, help="Image augmentation ratio (int)")

#     # model related arguments
#     # training related arguments
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--total_epochs", type=int, default=4000)
#     parser.add_argument("--batch_size", type=int, default=256)
#     parser.add_argument("--global_seed", type=int, default=42)
#     parser.add_argument("--num_workers", type=int, default=2, help="Number of GPUs to run the job")
#     parser.add_argument("--ckpt_every", type=int, default=25000, help="Number of iterations to save checkpoints.")

#     input_args = parser.parse_args()
#     input_args.data_path = '/blue/pinaki.sarder/j.fermin/Stem/hest_datasets/kidney/'
#     input_args.folder_list_filename = 'all_slide_lst.txt'
#     input_args.gene_list_filename = 'var_50genes.txt'
#     input_args.slide_out = 'NCBI703'
#     input_args.num_aug_ratio = 2
#     input_args.results_dir = "./kidney_results/CL/runs/"
    

#     print("mkdir & set up logger...")
#     # mkdir for logs and checkpoints
#     os.makedirs(input_args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
#     experiment_index = len(glob(f"{input_args.results_dir}/*"))
#     input_args.experiment_dir = f"{input_args.results_dir}/{experiment_index:03d}"  # Create an experiment folder
#     input_args.checkpoint_dir = f"{input_args.experiment_dir}/checkpoints"  # Stores saved model checkpoints
#     os.makedirs(input_args.checkpoint_dir, exist_ok=True)
#     os.makedirs(f"{input_args.experiment_dir}/samples", exist_ok=True)      # Store sampling results
#     input_args.logger = create_logger(input_args.experiment_dir)
#     input_args.logger.info(f"Experiment directory created at {input_args.experiment_dir}")

#     alldataset, input_args = assemble_dataset(input_args)

#     print(input_args)
#     print(alldataset.__len__())


#     loader, input_args = load_data(input_args)

#     for i, (gene, image, x_coord, y_coord) in enumerate(loader):
#         print(f"Batch {i+1}:")
#         print(f"  gene     shape: {gene.shape}")
#         print(f"  image    shape: {image.shape}")
#         print(f"  x_coord  shape: {x_coord.shape}")
#         print(f"  y_coord  shape: {y_coord.shape}")
#         print("-" * 40)
