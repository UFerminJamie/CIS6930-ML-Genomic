import argparse
import torch
import os
from CL.dataset import load_data, create_dir
from CL.model import mclSTExp_Attention
from tqdm import tqdm
from CL.utils import AvgMeter, get_lr



def train(model, train_dataLoader, optimizer, epoch):
    loss_meter = AvgMeter()
    tqdm_train = tqdm(train_dataLoader, total=len(train_dataLoader))
    for gene, image, x, y in tqdm_train:
        loss = model(gene.cuda(), image.cuda(), x.cuda(), y.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = image.size(0)
        loss_meter.update(loss.item(), count)
        tqdm_train.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer), epoch=epoch)


def save_model(args, model, epoch):
    torch.save(model.state_dict(),
                f"{args.checkpoint_dir}/CL_ckpt_{epoch}.pt")


def main(args):
    create_dir(args)
    train_dataLoader, args = load_data(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mclSTExp_Attention(spot_dim=args.dim,
                                temperature=args.temperature,
                                image_dim=args.cond_size,
                                projection_dim=args.projection_dim,
                                heads_num=args.heads_num,
                                heads_dim=args.heads_dim,
                                head_layers=args.heads_layers
                                )
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-3
    )
    for epoch in range(args.total_epochs):
        model.train()
        train(model, train_dataLoader, optimizer, epoch)
        save_model(args, model, epoch)
    print("Saved Model")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # data related arguments
    parser.add_argument("--expr_name", type=str, default="kidney")
    parser.add_argument("--data_path", type=str, default="/blue/pinaki.sarder/j.fermin/Stem/hest_datasets/kidney/", help="Dataset path")
    parser.add_argument("--results_dir", type=str, default="./kidney_results/CL/runs/", help="Path to hold runs")
    parser.add_argument("--slide_out", type=str, default="NCBI703,NCBI714,NCBI693", help="Test slide ID. Multiple slides separated by comma.") 
    parser.add_argument("--folder_list_filename", type=str, default="all_slide_lst.txt", help="A txt file listing file names for all training and testing slides in the dataset")
    parser.add_argument("--gene_list_filename", type=str, default="selected_gene_list.txt", help="Selected gene list")
    parser.add_argument("--num_aug_ratio", type=int, default=3, help="Image augmentation ratio (int)")
    # model related arguments
    parser.add_argument('--projection_dim', type=int, default=256, help='projection_dim')
    parser.add_argument('--heads_num', type=int, default=8, help='attention heads num')
    parser.add_argument('--heads_dim', type=int, default=64, help='attention heads dim')
    parser.add_argument('--dim', type=int, default=50, help='spot_embedding dimension (# HVGs)')  # 171, 785, 685
    parser.add_argument('--image_embedding_dim', type=int, default=1536, help='image_embedding dimension')
    parser.add_argument('--temperature', type=float, default=1., help='temperature')
    parser.add_argument('--heads_layers', type=int, default=2, help='attention heads layer num')
    # training related arguments
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    input_args = parser.parse_args()
    # input_args.data_path = '/blue/pinaki.sarder/j.fermin/Stem/hest_datasets/kidney/'
    # input_args.folder_list_filename = 'all_slide_lst.txt'
    # input_args.gene_list_filename = 'var_50genes.txt'
    # input_args.slide_out = 'NCBI703'
    # input_args.num_aug_ratio = 2
    # input_args.results_dir = "./kidney_results/CL/runs/"


    
    main(input_args)