import os
from os import listdir
from os.path import isfile, join
import math
import time
from tqdm import tqdm

import glob
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import scanpy as sc
import anndata

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from conch.open_clip_custom import create_model_from_pretrained
from huggingface_hub import login

import torchvision
from torchvision import datasets, models, transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

import argparse
from pathlib import Path

from typing import List
from loguru import logger



def get_img_patch_embd(img, 
                       adata,
                       samplename,
                       device,
                       save_path=None):
    
    

    check_file = os.path.join(save_path, "conch_ebd", samplename + "_conch.pt")
    if os.path.exists(check_file):
        print(f"[✔] Skipping {samplename}: Embeddings already exist.")
        return

    # CONCH model
    pretrained_CONCH, preprocess_CONCH = create_model_from_pretrained('conch_ViT-B-16', 
                                                                      "hf_hub:MahmoodLab/conch", 
                                                                      device=device,
                                                                      hf_auth_token="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"
                                                                      ) # TODO: need to replace "" by HuggingFace Login Token
    
    # UNI model
    login(token="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") # TODO: need to replace "" by HuggingFace Login Token
    timm_kwargs={"dynamic_img_size": True, "num_classes": 0, "init_values": 1.0}
    
    model_UNI = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs)
    transform_UNI = create_transform(**resolve_data_config(model_UNI.pretrained_cfg, model=model_UNI))

    def get_img_embd_conch(patch, model=pretrained_CONCH, preprocess=preprocess_CONCH):
        # resize to 256 by 256
        base_width = 256
        patch_resized = patch.resize((base_width, base_width), Image.Resampling.LANCZOS)
        patch_processed = preprocess(patch_resized).unsqueeze(0)
        with torch.inference_mode():
            feature_emb = model.encode_image(patch_processed.to(device), 
                                             proj_contrast=False, 
                                             normalize=False)        # [1, 512]
        return torch.clone(feature_emb)
    
    def get_img_embd_uni(patch, model=model_UNI, transform=transform_UNI):
        # resize to 224 by 224
        model.to(device)
        base_width = 224
        patch_resized = patch.resize((base_width, base_width), 
                                     Image.Resampling.LANCZOS)       # [224, 224]
        img_transformed = transform(patch_resized).unsqueeze(dim=0)  # [1, 3, 224, 224]
        with torch.inference_mode():
            feature_emb = model(img_transformed.to(device))          # [1, 1024]
        return torch.clone(feature_emb)

    def patch_augmentation_embd(patch, conch_or_uni,
                                num_transpose = 7):
        if conch_or_uni == "conch":
            embd_dim = 512
        elif conch_or_uni == "uni":
            embd_dim = 1024
        patch_aug_embd = torch.zeros(num_transpose, embd_dim)
        for trans in range(num_transpose):    # apply augmentations to the image patch
            patch_transposed = patch.transpose(trans)
            if conch_or_uni == "conch":
                patch_embd = get_img_embd_conch(patch_transposed)
            elif conch_or_uni == "uni":
                patch_embd = get_img_embd_uni(patch_transposed)
            patch_aug_embd[trans, :] = torch.clone(patch_embd)
        return patch_aug_embd.unsqueeze(0)


    # process spot
    spot_diameter = adata.uns["spatial"]["ST"]["scalefactors"]["spot_diameter_fullres"]
    print("Spot diameter: ", spot_diameter)  # Spot diameter for Visium
    if spot_diameter < 224: 
        radius = 112                         # minimum patch size: 224 by 224
    else:
        radius = int(spot_diameter // 2)
    x = adata.obsm["spatial"][:, 0]          # x coordinate in H&E image
    y = adata.obsm["spatial"][:, 1]          # y coordinate in H&E image

    all_patch_ebd_conch = None
    all_patch_ebd_uni = None
    all_patch_ebd_conch_aug = None
    all_patch_ebd_uni_aug = None
    first = True

    for spot_idx in tqdm(range(len(x))):
        patch = img.crop((x[spot_idx]-radius, y[spot_idx]-radius, 
                          x[spot_idx]+radius, y[spot_idx]+radius))
        patch_ebd_conch = get_img_embd_conch(patch)
        patch_ebd_uni   = get_img_embd_uni(patch)
        patch_ebd_conch_aug = patch_augmentation_embd(patch, "conch")
        patch_ebd_uni_aug   = patch_augmentation_embd(patch, "uni")

        if first:
            all_patch_ebd_conch = patch_ebd_conch
            all_patch_ebd_uni   = patch_ebd_uni
            all_patch_ebd_conch_aug = patch_ebd_conch_aug
            all_patch_ebd_uni_aug   = patch_ebd_uni_aug
            first = False
        else:
            all_patch_ebd_conch = torch.cat((all_patch_ebd_conch, patch_ebd_conch), dim=0)
            all_patch_ebd_uni   = torch.cat((all_patch_ebd_uni, patch_ebd_uni), dim=0)
            all_patch_ebd_conch_aug = torch.cat((all_patch_ebd_conch_aug, patch_ebd_conch_aug), dim=0)
            all_patch_ebd_uni_aug   = torch.cat((all_patch_ebd_uni_aug, patch_ebd_uni_aug), dim=0)
    print("Final data size: ", 
          all_patch_ebd_conch.shape, 
          all_patch_ebd_uni.shape,
          all_patch_ebd_conch_aug.shape,
          all_patch_ebd_uni_aug.shape)    

    if save_path != None:
        torch.save(all_patch_ebd_conch.detach().cpu(), os.path.join(save_path, "conch_ebd", samplename + "_conch.pt"))
        torch.save(all_patch_ebd_uni.detach().cpu(),   os.path.join(save_path, "uni_ebd", samplename + "_uni.pt"))
        torch.save(all_patch_ebd_conch_aug.detach().cpu(), os.path.join(save_path, "conch_ebd_aug", samplename + "_conch_aug.pt"))
        torch.save(all_patch_ebd_uni_aug.detach().cpu(),   os.path.join(save_path, "uni_ebd_aug", samplename + "_uni_aug.pt"))



def load_filtered_adata(st_path, fn_lst, verbose=False):
    """
    Load AnnData objects from a list of filenames, and retain only common genes.

    Args:
        st_path (str): Path to directory containing `.h5ad` files.
        fn_lst (List[str]): List of filenames (without `.h5ad` extension).
        verbose (bool): If True, prints progress and shapes.

    Returns:
        Tuple[List[AnnData], List[str], List[str]]:
            - Filtered list of AnnData objects with only common genes
            - Filtered list of filenames
            - Sorted list of common gene names
    """
    adata_lst = []
    first = True

    for fn in fn_lst:
        file_path = os.path.join(st_path, fn + ".h5ad")
        if not os.path.exists(file_path):
            if verbose:
                print(f"File not found: {file_path}. Skipping.")
            continue

        adata = anndata.read_h5ad(file_path)
        adata_lst.append(adata)

        if first:
            common_genes = adata.var_names
            first = False
            if verbose:
                print(f"First: {fn}, shape = {adata.shape}")
        else:
            common_genes = set(common_genes).intersection(set(adata.var_names))
            if verbose:
                print(f"{fn}: shape = {adata.shape}", end="\t")

    common_genes = sorted(list(common_genes))
    if verbose:
        print(f"\n Length of common genes: {len(common_genes)}")

    # Filter all adatas to keep only common genes
    for idx in range(len(fn_lst)):
        adata = adata_lst[idx].copy()
        adata_lst[idx] = adata[:, common_genes].copy()

        if verbose:
            print(f"{fn_lst[idx]} → {adata_lst[idx].shape}")

    if verbose:
        print("Only kept common genes across slides.")

    return adata_lst, fn_lst


def get_img_patch_embed_pt(st_path: str, fn_lst: List[str], data_path: str, device: str):

    os.makedirs(data_path + "/processed_data/", exist_ok=True)
    os.makedirs(data_path + "/processed_data/conch_ebd/", exist_ok=True)
    os.makedirs(data_path + "/processed_data/uni_ebd/", exist_ok=True)
    os.makedirs(data_path + "/processed_data/conch_ebd_aug/", exist_ok=True)
    os.makedirs(data_path + "/processed_data/uni_ebd_aug/", exist_ok=True)
    # if error, folder exists.
    
    adata_lst, fn_lst = load_filtered_adata(st_path, fn_lst)
    
    for i in range(len(fn_lst)):
        fn = fn_lst[i]
        adata = adata_lst[i]
        print(fn)
        image = Image.open(os.path.join(tif_path, fn + ".tif"))
        get_img_patch_embd(image, adata, fn, device, 
                           save_path=os.path.join(data_path,"processed_data"))
        print("#" * 20)

def get_hvg(adata_lst, fn_lst, verbose=True):
    union_hvg = set()

    for idx in range(len(fn_lst)):
        adata = adata_lst[idx].copy()
        fn = fn_lst[idx]
        
        sc.pp.filter_cells(adata, min_genes=1)
        sc.pp.filter_genes(adata, min_cells=1)
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000) # seuratv1

        union_hvg = union_hvg.union(set(adata.var_names[adata.var["highly_variable"]]))
        if verbose:
            print(fn, len(union_hvg))

    union_hvg = sorted([gene for gene in union_hvg if not gene.startswith(("MT", "mt", "RP", "rp"))])
    if verbose:
        print(f"Total number of unique highly variable genes (union across {len(fn_lst)} samples): {len(union_hvg)}")

    return union_hvg

def check_arg(arg, arg_name, values):
    if arg not in values:
        raise ValueError(f"{arg_name} can only be one of these: {values}, found {arg}")

def get_k_genes_from_df(args: argparse.Namespace, st_path: str, k: int, criteria: str, save_dir: str=None) -> List[str]:
    """Get the k genes according to some criteria across common genes in all the samples in the HEST meta dataframe

    Args:
        meta_df (pd.DataFrame): HEST meta dataframe
        k (int): number of genes to return
        criteria (str): criteria for the k genes to return
            - 'mean': return the k genes with the largest mean expression across samples
            - 'var': return the k genes with the largest expression variance across samples
        save_dir (str, optional): genes are saved as json array to this path if not None. Defaults to None.

    Returns:
        List[str]: k genes according to the criteria
    """

    st_lst = sorted(glob.glob(st_path + '/*'))
    adata_list = []
    for st_path in st_lst:
        adata = sc.read_h5ad(st_path)
        adata_list.append(adata)
    return get_k_genes(args, adata_list, k, criteria, save_dir=save_dir)


def get_k_genes(args: argparse.Namespace, adata_list: List[sc.AnnData], k: int, criteria: str, save_dir: str=None, min_cells_pct=0.10) -> List[str]: # type: ignore
    """Get the k genes according to some criteria across common genes in all the samples in the adata list

    Args:
        adata_list (List[sc.AnnData]): list of scanpy AnnData containing gene expressions in adata.to_df()
        k (int): number of most genes to return
        criteria (str): criteria for the k genes to return
            - 'mean': return the k genes with the largest mean expression across samples
            - 'var': return the k genes with the largest expression variance across samples
        save_dir (str, optional): genes are saved as json array to this path if not None. Defaults to None.
        min_cells_pct (float): filter out genes that are expressed in less than min_cells_pct% of the spots for each slide

    Returns:
        List[str]: k genes according to the criteria
    """
    import scanpy as sc
    
    check_arg(criteria, 'criteria', ['mean', 'var'])

    common_genes = None
    stacked_expressions = None

    # Get the common genes
    for adata in adata_list:
        my_adata = adata.copy()
        
        if min_cells_pct:
            print('min_cells is ', np.ceil(min_cells_pct * len(my_adata.obs)))
            sc.pp.filter_genes(my_adata, min_cells=np.ceil(min_cells_pct * len(my_adata.obs)))
        curr_genes = np.array(my_adata.to_df().columns)
        if common_genes is None:
            common_genes = curr_genes
        else:
            common_genes = np.intersect1d(common_genes, curr_genes)
            

    common_genes = [gene for gene in common_genes if 'BLANK' not in gene and 'Control' not in gene]
    logger.info(f"Found {len(common_genes)} common genes")

    for adata in adata_list:

        if stacked_expressions is None:
            stacked_expressions = adata.to_df()[common_genes]
        else:
            stacked_expressions = pd.concat([stacked_expressions, adata.to_df()[common_genes]])

    if criteria == 'mean':
        nb_spots = len(stacked_expressions)
        mean_expression = stacked_expressions.sum() / nb_spots
        
        top_k = mean_expression.nlargest(k).index
        save_dir = os.path.join(args.data_path, args.dataset_name, 'processed_data', f'mean_{k}genes.json')
        save_dir_txt = os.path.join(args.data_path, args.dataset_name, 'processed_data', f'mean_{k}genes.txt')
    elif criteria == 'var':
        stacked_adata = sc.AnnData(stacked_expressions.astype(np.float32))
        sc.pp.filter_genes(stacked_adata, min_cells=0)
        sc.pp.log1p(stacked_adata)
        sc.pp.highly_variable_genes(stacked_adata, n_top_genes=k)
        top_k = stacked_adata.var_names[stacked_adata.var['highly_variable']][:k].tolist()
        save_dir = os.path.join(args.data_path, args.dataset_name,'processed_data',f'var_{k}genes.json')
        save_dir_txt = os.path.join(args.data_path, args.dataset_name, 'processed_data', f'var_{k}genes.txt')

    else:
        raise NotImplementedError()

    if save_dir is not None:
        json_dict = {'genes': list(top_k)}
        with open(save_dir, mode='w') as json_file:
            json.dump(json_dict, json_file)

        with open(save_dir_txt, "w") as f:
            for gene in list(top_k):
                f.write(gene + "\n")

    logger.info(f'selected genes {top_k} based on {criteria}')
    return top_k





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='PRAD', help='Name of the dataset')
    parser.add_argument("--data_path", type=str, default='./hest1k_datasets', help="Base directory for dataset.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu).")
    parser.add_argument("--top_k", type=int, default=50, help="Number of genes to be selected.")
    parser.add_argument("--criteria", type=str, nargs='+', default=['var', 'mean'], help="Criteria in the gene selection.")
    parser.add_argument("--generate_embedding", type=bool, default=False, help="Generate patch embeddings.")
    args = parser.parse_args()

    data_path = os.path.join(args.data_path, args.dataset_name)
    tif_path = data_path + "/wsis"
    st_path = data_path +  "/st"
    
    if args.dataset_name == 'PRAD':
        fn_lst = sorted(os.listdir(tif_path))
        fn_lst = [os.path.splitext(f)[0] for f in fn_lst]
    elif args.dataset_name == 'her2st':
        fn_lst = sorted(os.listdir(tif_path))
        fn_lst = [os.path.splitext(f)[0] for f in fn_lst]
    elif args.dataset_name == 'kidney':
        fn_lst = sorted(os.listdir(tif_path))
        fn_lst = [os.path.splitext(f)[0] for f in fn_lst]
    elif args.dataset_name == 'mouse_brain':
        fn_lst = sorted(os.listdir(tif_path))
        fn_lst = [os.path.splitext(f)[0] for f in fn_lst]
        try:
            fn_lst.remove("NCBI653")
            fn_lst.remove("NCBI654") 
            fn_lst.remove("NCBI655") 
            fn_lst.remove("NCBI656")  
            fn_lst.remove("NCBI657") 
        except:
            pass
    else:
        raise ValueError(f"Dataset '{args.dataset_name}' is not supported. Please choose from ['PRAD', 'her2st', 'kidney', 'mouse_brain'].")
   
    filepath = os.path.join(data_path, 'processed_data','all_slide_lst.txt')
    
    with open(filepath, "w") as f:
        for slide_name in fn_lst:
            f.write(slide_name + "\n")

    if args.generate_embedding:
        get_img_patch_embed_pt(st_path, fn_lst, data_path, args.device)

    for criterion in args.criteria:
        get_k_genes_from_df(args, st_path, args.top_k, criterion, data_path)