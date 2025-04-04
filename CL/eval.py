from dataset import load_data, create_save_dir
import torch
import numpy as np
from model import MultiModal
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn.functional as F
import os

def get_embeddings(model, input_args):
    test_loader, input_args = load_data(input_args)

    state_dict = torch.load(input_args.model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Finished loading model")

    test_image_embeddings = []
    spot_embeddings = []
    gene_ori = []
    with torch.no_grad():
        for gene, image, x, y in tqdm(test_loader):
            gene_ori.append(gene)
            image_embeddings = model.image_projection(image.cuda())
            test_image_embeddings.append(image_embeddings)

            spot_feature = gene.cuda()
            x = x.long().cuda()
            y = y.long().cuda()
            centers_x = model.x_embed(x)
            centers_y = model.y_embed(y)
            spot_feature = spot_feature + centers_x + centers_y

            spot_features = spot_feature.unsqueeze(dim=0)
            spot_embedding = model.spot_encoder(spot_features)
            spot_embedding = model.spot_projection(spot_embedding).squeeze(dim=0)
            spot_embeddings.append(spot_embedding)
    return torch.cat(test_image_embeddings), torch.cat(spot_embeddings), torch.cat(gene_ori)


def save_embeddings(args):
    model = MultiModal(spot_dim=args.dim,
                               temperature=args.temperature,
                               image_dim=args.image_embedding_dim,
                               projection_dim=args.projection_dim,
                               heads_num=args.heads_num,
                               heads_dim=args.heads_dim,
                               head_layers=args.heads_layers
                               ).to('cuda')

    img_embeddings_all, spot_embeddings_all, gene_ori_all = get_embeddings(model, args)

    img_embeddings_all = img_embeddings_all.cpu().numpy()
    spot_embeddings_all = spot_embeddings_all.cpu().numpy()
    gene_ori_all = gene_ori_all.cpu().numpy()
    # print(img_embeddings_all.shape)
    # print(spot_embeddings_all.shape)
    # print(gene_ori_all.shape)

    if args.test:
        index_start = 0
        print(args.datasize)
        slidename = list(args.slide_out.split(','))
        print(args.slide_out)
        for size, name in zip(args.datasize, slidename):
            index_end = index_start + size
            image_embeddings = img_embeddings_all[index_start:index_end]
            spot_embeddings = spot_embeddings_all[index_start:index_end]
            gene_ori = gene_ori_all[index_start:index_end]

            # print(image_embeddings.shape)
            # print(spot_embeddings.shape)
            # print(gene_ori.shape)

            np.save(args.save_path + '/embeddings/' + f"{name}_img_embeddings.npy", image_embeddings.T)
            np.save(args.save_path + '/embeddings/' + f"{name}_spot_embeddings.npy", spot_embeddings.T)
            np.save(args.save_path + '/embeddings/' + f"{name}_original_gene_expressions.npy", gene_ori)
            index_start = index_end 

    else:
        np.save(args.save_path + '/embeddings/' + "img_embeddings.npy", img_embeddings_all.T)
        np.save(args.save_path + '/embeddings/' + "spot_embeddings.npy", spot_embeddings_all.T)
        np.save(args.save_path + '/embeddings/' + "original_gene_expressions.npy", gene_ori_all)



def find_matches(spot_embeddings, query_embeddings, top_k=1):
    """
    For each query embedding, find top_k most similar embeddings in spot_embeddings
    using cosine similarity.
    Args:
        spot_embeddings: (N, D)
        query_embeddings: (M, D)
        top_k: int, number of top matches to return
    Returns:
        indices: (M, top_k) array of indices of top_k matches in spot_embeddings for each query
    """
    spot_embeddings = torch.tensor(spot_embeddings, dtype=torch.float32)
    query_embeddings = torch.tensor(query_embeddings, dtype=torch.float32)
    # L2 normalize
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    # Compute cosine similarity
    dot_similarity = query_embeddings @ spot_embeddings.T  # (M, N)
    # Top-k indices for each query
    _, topk_indices = torch.topk(dot_similarity, k=top_k, dim=1)

    return topk_indices.cpu().numpy()  # shape: (M, top_k)

def mean_pcc(true, pred):
    """
    Compute the mean PCC over all samples.
    true, pred: numpy arrays of shape (N, D)
    """
    pccs = []
    for i in range(true.shape[1]):
        pcc = np.corrcoef(true[:,i], pred[:,i])[0, 1]
        if not np.isnan(pcc):
            pccs.append(pcc)
    
    return pccs

def RVD(pred_avg,gt):
    pred_var = np.var(pred_avg, axis=0)
    gt_var = np.var(gt, axis=0)
    print("RVD: ", np.mean((pred_var - gt_var)**2 / gt_var**2))

def metrics(true, pred, input_args):
    mse = mean_squared_error(true, pred)
    print("Mean Squared Error (MSE): ", mse)
    mae = mean_absolute_error(true, pred)
    print("Mean Absolute Error (MAE): ", mae)
    pcc = mean_pcc(true, pred)
    avg_pcc = np.mean(pcc)
    print("Mean PCC:", avg_pcc) 
    print("PCC-10: ", np.mean(sorted(pcc)[::-1][:10]))
    print("PCC-50: ", np.mean(sorted(pcc)[::-1][:50]))
    print("PCC-200: ", np.mean(sorted(pcc)[::-1][:200]))


    selected_genes = list(np.genfromtxt(input_args.data_path + "processed_data/" + input_args.gene_list_filename, dtype=str))
    paired = list(zip(pcc, selected_genes))
    paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
    sorted_pcc, sorted_genes = zip(*paired_sorted)
    sorted_genes = list(sorted_genes)

    # RVD(pred,true)
    if input_args.all_preds:
        output_path = input_args.save_path + '/results/' + f"result.txt"
    else:
        output_path = input_args.save_path + '/results/' + f"{input_args.name}.txt"
       
    with open(output_path, "w") as f:
        f.write("pcc,gene\n")
        for score, gene in zip(sorted_pcc, sorted_genes):
            f.write(f"{score:.6f},{gene}\n")
    


def generate_prediction(args):
    slidename = list(args.slide_out.split(','))

    spot_embeddings = np.load(args.save_path + "/embeddings/" + "spot_embeddings.npy")
    expression_gt = np.load(args.save_path + "/embeddings/" + "original_gene_expressions.npy")
    all_preds = []
    all_refs = []
    for name in slidename:
        print(f'Doing Prediction on {name}')
        input_args.name=name
        image_query = np.load(args.save_path + "/embeddings/" + f"{name}_img_embeddings.npy")
        reference = np.load(args.save_path + "/embeddings/" + f"{name}_original_gene_expressions.npy")

        if image_query.shape[1] != 256:
            image_query = image_query.T
            print("image query shape: ", image_query.shape)
        if expression_gt.shape[1] != 200:
            expression_gt = expression_gt.T
            print("expression_gt shape: ", expression_gt.shape)
        if spot_embeddings.shape[1] != 256:
            spot_embeddings = spot_embeddings.T
            print("spot_embeddings shape: ", spot_embeddings.shape)
        if reference.shape[0] != image_query.shape[0]:
            reference = reference.T
            print("reference shape: ", reference.shape)
        
        indices = find_matches(spot_embeddings, image_query, args.top_k)
        matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_embeddings.shape[1]))
        matched_spot_expression_pred = np.zeros((indices.shape[0], expression_gt.shape[1]))
        mean_spot_expression_pred = np.zeros((indices.shape[0], expression_gt.shape[1]))
        # median_spot_expression_pred = np.zeros((indices.shape[0], expression_gt.shape[1]))
        all_refs.append(reference)
        for i in range(indices.shape[0]):
            a = np.linalg.norm(spot_embeddings[indices[i, :], :] - image_query[i, :], axis=1)

            reciprocal_of_square_a = np.reciprocal(a ** 2)
            weights = reciprocal_of_square_a / np.sum(reciprocal_of_square_a)
            weights = weights.flatten()
            matched_spot_embeddings_pred[i, :] = np.average(spot_embeddings[indices[i, :], :], axis=0, weights=weights)
            matched_spot_expression_pred[i, :] = np.average(expression_gt[indices[i, :], :], axis=0,
                                                            weights=weights)
            mean_spot_expression_pred[i, :] = np.mean(expression_gt[indices[i, :], :], axis=0)
        all_preds.append(mean_spot_expression_pred)    

            # median_spot_expression_pred[i, :] = np.median(expression_gt[indices[i, :], :], axis=0)

        # print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
        # print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)
        # print("mean spot embeddings pred shape: ", mean_spot_expression_pred.shape)
        # print("median spot expression pred shape: ", median_spot_expression_pred.shape)
        # print("mode spot expression pred shape: ", mode_spot_expression_pred.shape)
        os.makedirs(args.save_path + "/results/", exist_ok=True)
        np.save(args.save_path + "/results/" + f"{name}_pred.npy", matched_spot_expression_pred)
        np.save(args.save_path + "/results/" + f"{name}_pred_mean.npy", mean_spot_expression_pred)
        # np.save(args.save_path + "/embeddings/" + f"{name}_pred_median.npy", median_spot_expression_pred)

        all_preds_concat = np.concatenate(all_preds, axis=0)
        all_refs_concat = np.concatenate(all_refs, axis=0)
        # performance metric
        # print("================WEIGHTED PREDICTIONS==================")
        # metrics(reference, matched_spot_expression_pred, args)
        print("================PREDICTIONS======================")
        args.all_preds=False
        metrics(reference, mean_spot_expression_pred, args)
    args.all_preds=True
    metrics(all_refs_concat, all_preds_concat, args)

def main(args):
    create_save_dir(args)
    # get embeddings from train samples
    args.test = False
    save_embeddings(args)
    # get embeddings from test samples
    args.test = True
    save_embeddings(args)

    generate_prediction(args)





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr_name", type=str, default="kidney")
    parser.add_argument("--data_path", type=str, default=f'./hest_datasets/kidney/', help="Dataset path")
    # parser.add_argument("--save_path", type=str, default="/blue/pinaki.sarder/j.fermin/Stem/kidney_results/CL/runs/006/", help="Path to hold runs")
    parser.add_argument("--slide_out", type=str, default="NCBI703,NCBI714,NCBI693", help="Test slide ID. Multiple slides separated by comma.") 
    parser.add_argument("--folder_list_filename", type=str, default="all_slide_lst.txt", help="A txt file listing file names for all training and testing slides in the dataset")
    parser.add_argument("--gene_list_filename", type=str, default='var_200genes.txt', help="Selected gene list")
    # parser.add_argument("--model_path", type=str, default="/blue/pinaki.sarder/j.fermin/Stem/kidney_results/CL/runs/006/checkpoints/CL_ckpt_199.pt", help="model checkpoint")
    parser.add_argument("--num_aug_ratio", type=int, default=0, help="number of augmentations")
    parser.add_argument("--test", type=bool, default=False, help="when true generate test embeddings")
    parser.add_argument("--top_k", type=int, default=100, help="when true generate test embeddings")
    # model related arguments
    parser.add_argument('--projection_dim', type=int, default=256, help='projection_dim')
    parser.add_argument('--heads_num', type=int, default=8, help='attention heads num')
    parser.add_argument('--heads_dim', type=int, default=64, help='attention heads dim')
    parser.add_argument('--dim', type=int, default=200, help='spot_embedding dimension (# HVGs)')
    parser.add_argument('--image_embedding_dim', type=int, default=1536, help='image_embedding dimension')
    parser.add_argument('--temperature', type=float, default=1., help='temperature')
    parser.add_argument('--heads_layers', type=int, default=2, help='attention heads layer num')

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--run", type=str, default="")
    input_args = parser.parse_args()
    # expr_name = 'kidney'
    # runs = '011'
    # input_args.slide_out = "NCBI703,NCBI714,NCBI693"
    # expr_name = 'her2st'
    # runs = '027'
    # input_args.slide_out = "SPA148,SPA136"
    # input_args.gene_list_filename='svg_200genes.txt'
    expr_name=input_args.expr_name
    runs=input_args.run

    input_args.data_path = f'./hest_datasets/{expr_name}/'
    input_args.model_path = f"./{expr_name}_results/CL/runs/{runs}/checkpoints/CL_ckpt_199.pt"
    input_args.save_path = f"./{expr_name}_results/CL/runs/{runs}/"
    
    main(input_args)