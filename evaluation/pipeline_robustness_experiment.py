import numpy as np

import torch
from torch.utils.data import DataLoader

from models.cell_retrieval import CellRetrievalNetwork
from models.cross_matcher import CrossMatch

from evaluation.args import parse_arguments
from evaluation.utils import calc_sample_accuracies, print_accuracies

from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti

from datapreparation.kitti360pose.utils import KNOWN_CLASS
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360

from supplementary_experiment import eval_epoch as eval_epoch_retrieval

import torch_geometric.transforms as T





@torch.no_grad()
def run_coarse(model, dataloader, args, thresh_query, thresh_database):
    """Run text-to-cell retrieval to obtain the top-cells and coarse pose accuracies.

    Args:
        model: retrieval model
        dataloader: retrieval dataset
        args: global arguments

    Returns:
        [List]: retrievals as [(cell_indices_i_0, cell_indices_i_1, ...), (cell_indices_i+1, ...), ...] with i ∈ [0, len(poses)-1], j ∈ [0, max(top_k)-1]
        [Dict]: accuracies
    """
    model.eval()

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

        # Run retrieval model to obtain top-cells
    
    #Fix the query set size and continuously increase the database size
    retrieval_accuracies, retrieval_accuracies_close, retrievals = eval_epoch_retrieval(
        model, dataloader, args, thresh_query, thresh_database
    )
    retrievals = [retrievals[idx] for idx in range(len(retrievals))]  # Dict -> list
    print("Retrieval Accs:")
    print(retrieval_accuracies)
    print("Retrieval Accs Close:")
    print(retrieval_accuracies_close)

    # Gather the accuracies for each sample
    accuracies = {k: {t: [] for t in args.threshs} for k in args.top_k}
    for i_sample in range(len(retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        pos_in_cells = 0.5 * np.ones((len(top_cells), 2))  # Predict cell-centers

        accs = calc_sample_accuracies(pose, top_cells, pos_in_cells, args.top_k, args.threshs)

        for k in args.top_k:
            for t in args.threshs:
                accuracies[k][t].append(accs[k][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies[k][t] = np.mean(accuracies[k][t])

    return retrievals, accuracies



if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))

    # Load datasets
    if args.no_pc_augment:
        transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    if args.no_pc_augment_fine:
        transform_fine = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform_fine = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    SCENE_NAMES_SUP=["2013_05_28_drive_0009_sync",]
    dataset_retrieval = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_SUP, transform, shuffle_hints=False, flip_poses=False,)
    
    dataloader_retrieval = DataLoader(
        dataset_retrieval,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    # Load models
    model_coarse_dic = torch.load(args.path_coarse, map_location=torch.device("cpu"))
    model_coarse = CellRetrievalNetwork(
                KNOWN_CLASS,
                COLOR_NAMES_K360,
                args,
            )
    model_coarse.load_state_dict(model_coarse_dic, strict = False)
    model_coarse.to(device)

    if args.path_fine:
        model_fine_dic = torch.load(args.path_fine, map_location=torch.device("cpu"))
        model_fine = CrossMatch(
            KNOWN_CLASS,
            COLOR_NAMES_K360,
            args,
        )
        model_fine.load_state_dict(model_fine_dic, strict = False)
        model_fine.to(device)


    # Run coarse
    thresh_query=200
    thresh_database=200
    retrievals, coarse_accuracies = run_coarse(model_coarse, dataloader_retrieval, args ,thresh_query,thresh_database)
    print_accuracies(coarse_accuracies, "Coarse")
