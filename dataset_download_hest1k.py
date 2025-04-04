from huggingface_hub import login
import datasets
import os
import pandas as pd
import argparse
from typing import List




def download_dataset(dataset_title: str, data_path:str):
    login(token="") # token=Your_HuggingFace_Token
    
    # load metadata
    meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")

    if dataset_title == 'PRAD':
        ids_to_query = ["MEND"+str(i) for i in range(139, 163)]
    else:
        ids_to_query = meta_df.loc[meta_df["dataset_title"].str.startswith(dataset_title), "id"].tolist()
        
    list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
    dataset = datasets.load_dataset(
        'MahmoodLab/hest', 
        cache_dir=data_path,
        patterns=list_patterns
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='PRAD', help='Name of the dataset')
    parser.add_argument("--data_path", type=str, default='./hest_datasets', help="Base directory for dataset.")
    args = parser.parse_args()

    data_path = os.path.join(args.data_path, args.dataset_name)
    os.makedirs(data_path, exist_ok=True)

    if args.dataset_name == 'kidney':
        dataset_title = "Spatial localization with Spatial Transcriptomics for an atlas of healthy and injured cell states"
        download_dataset(dataset_title, data_path)
    elif args.dataset_name == 'her2st':
        dataset_title = "Spatial deconvolution of"
        download_dataset(dataset_title, data_path)
    elif args.dataset_name == 'PRAD':
        dataset_title = args.dataset_name
        download_dataset(dataset_title, data_path)
    elif args.dataset_name == 'mouse_brain':
        dataset_title = "Spatial Multimodal Analysis"
        download_dataset(dataset_title, data_path)

    else:
        raise ValueError(f"Dataset '{args.dataset_name}' is not supported. Please choose from ['PRAD', 'her2st', 'kidney', 'mouse_brain'].")
