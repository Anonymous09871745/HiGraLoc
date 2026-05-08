"""Module for training the coarse cell-retrieval module
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch_geometric.transforms as T
import collections
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easydict import EasyDict
import os
import os.path as osp
import tqdm

from models.cell_retrieval import CellRetrievalNetwork

from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti, Kitti360CoarseDataset

from training.args import parse_arguments
from training.losses import PairwiseRankingLoss, HardestRankingLoss, ContrastiveLoss, CCL, CCL_input_score, ContrastiveLoss_input_score


def z_score(tensor):
    # input：[Batch_size,Batch_size]
    mean = np.mean(tensor)
    std = np.std(tensor)
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor


def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []

    batches = []
    pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    thresh = 0.8

    for i_batch, batch in pbar:
        optimizer.zero_grad()

        anchor, F_Object_T, F_Relation_T = model.encode_text(batch["texts"])
        F_Global, F_Object_P, F_Relation_P, p_object_level_masks, p_relation_level_masks = model.encode_objects(
            batch["objects"], batch["object_points"])
        batch_size = len(anchor)

        # calculate similarity score of Object-level
        aa_expanded = F_Object_T.unsqueeze(
            1).expand(batch_size, batch_size, -1, -1)
        bb_transposed = F_Object_P.transpose(1, 2)
        cc = torch.matmul(aa_expanded, bb_transposed.unsqueeze(0)).squeeze(0)
        aa_mask = torch.ones(
            (batch_size, batch_size, F_Object_T.size(1), 1)).to('cuda')
        bb_mask = p_object_level_masks.transpose(1, 2)
        score_mask = torch.matmul(aa_mask, bb_mask)
        cc[score_mask == 0] = -100
        score = cc.max(dim=-1)[0].mean(dim=-1)
        mean_F_Object_P = F_Object_P.mean(dim=1)
        mean_F_Object_P = mean_F_Object_P / \
            torch.norm(mean_F_Object_P, p=2, dim=1, keepdim=True)
        mean_F_Object_T = F_Object_T.mean(dim=1)
        mean_F_Object_T = mean_F_Object_T / \
            torch.norm(mean_F_Object_T, p=2, dim=1, keepdim=True)
        dist_Object = abs(torch.norm(mean_F_Object_P.unsqueeze(
            1)-mean_F_Object_T.unsqueeze(0), dim=2))
        dist_Object = dist_Object/dist_Object.max()
        dist_Object = pow((1-dist_Object+1e-6), 1/args.alpha)
        dist_Object[dist_Object > 1] = 1
        dist_Object[dist_Object < thresh] = thresh

        # print(dist_Object.mean())

        # calculate similarity score of Relation-level
        aa_expanded = F_Relation_T.unsqueeze(
            1).expand(batch_size, batch_size, -1, -1)
        bb_transposed = F_Relation_P.transpose(1, 2)
        cc = torch.matmul(aa_expanded, bb_transposed.unsqueeze(0)).squeeze(0)
        aa_mask = torch.ones(
            (batch_size, batch_size, F_Relation_T.size(1), 1)).to('cuda')
        bb_mask = p_relation_level_masks.transpose(1, 2)
        score_mask = torch.matmul(aa_mask, bb_mask)
        cc[score_mask == 0] = -100
        relation_score = cc.max(dim=-1)[0].mean(dim=-1)
        mean_F_Relation_P = F_Relation_P.mean(dim=1)
        mean_F_Relation_P = mean_F_Relation_P / \
            torch.norm(mean_F_Relation_P, p=2, dim=1, keepdim=True)
        mean_F_Relation_T = F_Relation_T.mean(dim=1)
        mean_F_Relation_T = mean_F_Relation_T / \
            torch.norm(mean_F_Relation_T, p=2, dim=1, keepdim=True)
        dist_Relation = abs(torch.norm(mean_F_Relation_P.unsqueeze(
            1)-mean_F_Relation_T.unsqueeze(0), dim=2))
        dist_Relation = dist_Relation/dist_Relation.max()
        dist_Relation = pow((1-dist_Relation+1e-6), 1/args.alpha)
        dist_Relation[dist_Relation > 1] = 1
        dist_Relation[dist_Relation < thresh] = thresh
        # print(dist_Relation.mean())

        # calculate similarity score of Global-level
        # loss = 1.0*criterion(anchor, F_Global)+1.0*criterion_2(score,dist_Object)+1.0*criterion_2(relation_score, dist_Relation)+1.0*criterion(anchor, anchor) 配合./checkpoints/k360_30-10_scG_pd10_pc4_spY_all/Text_Constructive/coarse_contN_epoch24_acc0.880_ecl0_eco0_p256_npa1_loss-CCL_f-class-color-position-num.pth 权重实现了普遍的0.004提高
            
        # 三分支权重 (Global, Object, Relation) = (0.5, 1.5, 1.0)
        loss = 0.5*criterion(anchor, F_Global)+1.5*criterion_2(score,
                                                               dist_Object)+1.0*criterion_2(relation_score, dist_Relation)

        if (torch.isnan(loss).any()):
            import ipdb
            ipdb.set_trace()

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        torch.cuda.empty_cache()

    return np.mean(epoch_losses), batches


@torch.no_grad()
def eval_epoch(model, dataloader, args, return_encodings=False, return_distance=False):

    model.eval()
    accuracies = {k: [] for k in args.top_k}
    accuracies_close = {k: [] for k in args.top_k}

    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size

    cell_encodings_global_level = np.zeros(
        (len(cells_dataset), model.embed_dim))
    cell_encodings_object_level = np.zeros(
        (len(cells_dataset), args.object_size, model.embed_dim))
    cell_encodings_relation_level = np.zeros(
        (len(cells_dataset), args.object_size, model.embed_dim))
    cell_encodings_object_level_mask = np.zeros(
        (len(cells_dataset), args.object_size, 1))
    cell_encodings_relation_level_mask = np.zeros(
        (len(cells_dataset), args.object_size, 1))

    db_cell_ids = np.zeros(len(cells_dataset), dtype="<U32")

    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))
    text_encodings_object_level = np.zeros(
        (len(dataloader.dataset), args.num_mentioned, model.embed_dim))
    text_encodings_relation_level = np.zeros(
        (len(dataloader.dataset), args.num_mentioned, model.embed_dim))
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype="<U32")
    query_poses_w = np.array([pose.pose_w[0:2]
                             for pose in dataloader.dataset.all_poses])

    # Encode the query side
    t0 = time.time()
    index_offset = 0
    for batch in tqdm.tqdm(dataloader):
        text_enc, text_enc_object_level, text_enc_relation_level = model.encode_text(
            batch["texts"])
        batch_size = len(text_enc)

        text_encodings[index_offset: index_offset + batch_size, :] = (
            text_enc.cpu().detach().numpy()
        )
        text_encodings_object_level[index_offset: index_offset + batch_size, :, :] = (
            text_enc_object_level.cpu().detach().numpy()
        )
        text_encodings_relation_level[index_offset: index_offset + batch_size, :, :] = (
            text_enc_relation_level.cpu().detach().numpy()
        )
        query_cell_ids[index_offset: index_offset +
                       batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size
        del text_enc, text_enc_object_level, text_enc_relation_level
    print(
        f"Encoded {len(text_encodings)} query texts in {time.time() - t0:0.2f}.")

    # Encode the database side
    index_offset = 0
    for batch in cells_dataloader:
        cell_enc_global_level, cell_enc_object_level, cell_enc_relation_level, cell_enc_object_level_mask, cell_enc_relation_level_mask = model.encode_objects(
            batch["objects"], batch["object_points"])
        batch_size = len(cell_enc_object_level)

        cell_encodings_global_level[index_offset: index_offset + batch_size, :] = (
            cell_enc_global_level.cpu().detach().numpy()
        )

        cell_encodings_object_level[index_offset: index_offset + batch_size, :, :] = (
            cell_enc_object_level.cpu().detach().numpy()
        )

        cell_encodings_relation_level[index_offset: index_offset + batch_size, :, :] = (
            cell_enc_relation_level.cpu().detach().numpy()
        )

        cell_encodings_object_level_mask[index_offset: index_offset + batch_size, :, :] = (
            cell_enc_object_level_mask.cpu().detach().numpy()
        )

        cell_encodings_relation_level_mask[index_offset: index_offset + batch_size, :, :] = (
            cell_enc_relation_level_mask.cpu().detach().numpy()
        )
        db_cell_ids[index_offset: index_offset +
                    batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size
        del cell_enc_global_level, cell_enc_object_level, cell_enc_relation_level, cell_enc_object_level_mask, cell_enc_relation_level_mask

    top_retrievals = {}  # {query_idx: top_cell_ids}
    if return_distance:
        dists_list = []
        scores_list = []
    for query_idx in range(len(text_encodings)):
        if args.ranking_loss != "triplet":  # Asserted above
            # similarity score of Global-level
            scores_1 = cell_encodings_global_level[:] @ text_encodings[query_idx]

            # similarity score of Object-level
            aa = torch.from_numpy(text_encodings_object_level[query_idx])
            bb = torch.from_numpy(cell_encodings_object_level)
            aa = aa.unsqueeze(0).expand(
                len(cell_encodings_object_level), args.num_mentioned, args.coarse_embed_dim)
            bb = bb.transpose(1, 2)
            cc = torch.matmul(aa, bb)
            aa_mask = torch.ones((aa.size(1), 1)).to(
                dtype=torch.float64).to('cuda')
            bb_mask = torch.from_numpy(
                cell_encodings_object_level_mask).transpose(1, 2).to('cuda')
            scores_2_mask = torch.matmul(aa_mask, bb_mask)
            cc[scores_2_mask == 0] = -100
            scores_2 = cc.max(dim=-1)[0].mean(dim=-1)

            # similarity score of Relation-level
            aa = torch.from_numpy(text_encodings_relation_level[query_idx])
            bb = torch.from_numpy(cell_encodings_relation_level)
            aa = aa.unsqueeze(0).expand(
                len(cell_encodings_object_level), args.num_mentioned, args.coarse_embed_dim)
            bb = bb.transpose(1, 2)
            cc = torch.matmul(aa, bb)
            aa_mask = torch.ones((aa.size(1), 1)).to(
                dtype=torch.float64).to('cuda')
            bb_mask = torch.from_numpy(
                cell_encodings_relation_level_mask).transpose(1, 2).to('cuda')
            scores_3_mask = torch.matmul(aa_mask, bb_mask)
            cc[scores_3_mask == 0] = -100
            scores_3 = cc.max(dim=-1)[0].mean(dim=-1)

            # 分支权重 (w1, w2, w3) = (0.5, 1.5, 1.0)
            scores = 0.5 * z_score(scores_1) + 1.5 * z_score(np.array(scores_2)) + 1.0 * z_score(np.array(scores_3))

            assert len(scores) == len(dataloader.dataset.all_cells)
            sorted_indices = np.argsort(-1.0 * scores)  # High -> low

        sorted_indices = sorted_indices[0: np.max(args.top_k)]

        # Best-cell hit accuracy
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        target_cell_id = query_cell_ids[query_idx]

        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[0:k])
        top_retrievals[query_idx] = retrieved_cell_ids

        # Close-by accuracy
        # CARE/TODO: can be wrong across scenes!
        target_pose_w = query_poses_w[query_idx]
        retrieved_cell_poses = [
            cells_dict[cell_id].get_center()[0:2] for cell_id in retrieved_cell_ids
        ]
        dists = np.linalg.norm(target_pose_w - retrieved_cell_poses, axis=1)
        if return_distance:
            dists_list.append(dists[0:max(args.top_k)])
            scores_list.append(scores[sorted_indices])
        for k in args.top_k:
            accuracies_close[k].append(np.any(dists[0:k] <= cell_size / 2))

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
        accuracies_close[k] = np.mean(accuracies_close[k])

    if return_encodings:
        return accuracies, accuracies_close, top_retrievals, cell_encodings, text_encodings
    elif return_distance:
        return accuracies, accuracies_close, top_retrievals, cell_encodings, text_encodings, np.stack(dists_list), np.stack(scores_list)
    else:
        return accuracies, accuracies_close, top_retrievals


@torch.no_grad()
def eval_epoch_return_score_matrices(model, dataloader, args):
    """与 eval_epoch 完全相同的编码与打分逻辑，但返回 (S1, S2, S3) 供分支权重搜索用。
    返回: S1, S2, S3 各 (n_queries, n_cells), query_cell_ids, db_cell_ids, top_k_list
    """
    model.eval()
    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    n_queries = len(dataloader.dataset)
    n_cells = len(cells_dataset)

    cell_encodings_global_level = np.zeros((n_cells, model.embed_dim))
    cell_encodings_object_level = np.zeros((n_cells, args.object_size, model.embed_dim))
    cell_encodings_relation_level = np.zeros((n_cells, args.object_size, model.embed_dim))
    cell_encodings_object_level_mask = np.zeros((n_cells, args.object_size, 1))
    cell_encodings_relation_level_mask = np.zeros((n_cells, args.object_size, 1))
    db_cell_ids = np.zeros(n_cells, dtype="<U32")

    text_encodings = np.zeros((n_queries, model.embed_dim))
    text_encodings_object_level = np.zeros((n_queries, args.num_mentioned, model.embed_dim))
    text_encodings_relation_level = np.zeros((n_queries, args.num_mentioned, model.embed_dim))
    query_cell_ids = np.zeros(n_queries, dtype="<U32")

    index_offset = 0
    for batch in tqdm.tqdm(dataloader, desc="Encode queries"):
        text_enc, text_enc_object_level, text_enc_relation_level = model.encode_text(batch["texts"])
        bs = len(text_enc)
        text_encodings[index_offset:index_offset + bs] = text_enc.cpu().numpy()
        text_encodings_object_level[index_offset:index_offset + bs] = text_enc_object_level.cpu().numpy()
        text_encodings_relation_level[index_offset:index_offset + bs] = text_enc_relation_level.cpu().numpy()
        query_cell_ids[index_offset:index_offset + bs] = np.array(batch["cell_ids"])
        index_offset += bs

    index_offset = 0
    for batch in tqdm.tqdm(cells_dataloader, desc="Encode cells"):
        c_global, c_obj, c_rel, m_obj, m_rel = model.encode_objects(batch["objects"], batch["object_points"])
        bs = len(c_global)
        cell_encodings_global_level[index_offset:index_offset + bs] = c_global.cpu().numpy()
        cell_encodings_object_level[index_offset:index_offset + bs] = c_obj.cpu().numpy()
        cell_encodings_relation_level[index_offset:index_offset + bs] = c_rel.cpu().numpy()
        cell_encodings_object_level_mask[index_offset:index_offset + bs] = m_obj.cpu().numpy()
        cell_encodings_relation_level_mask[index_offset:index_offset + bs] = m_rel.cpu().numpy()
        db_cell_ids[index_offset:index_offset + bs] = np.array(batch["cell_ids"])
        index_offset += bs

    device = next(model.parameters()).device
    S1 = np.zeros((n_queries, n_cells), dtype=np.float32)
    S2 = np.zeros((n_queries, n_cells), dtype=np.float32)
    S3 = np.zeros((n_queries, n_cells), dtype=np.float32)

    for query_idx in tqdm.tqdm(range(n_queries), desc="Build score matrices"):
        scores_1 = cell_encodings_global_level[:] @ text_encodings[query_idx]
        S1[query_idx] = scores_1

        aa = torch.from_numpy(text_encodings_object_level[query_idx]).to(device)
        bb = torch.from_numpy(cell_encodings_object_level).to(device)
        aa = aa.unsqueeze(0).expand(n_cells, args.num_mentioned, args.coarse_embed_dim)
        bb = bb.transpose(1, 2)
        cc = torch.matmul(aa, bb)
        aa_mask = torch.ones((args.num_mentioned, 1), dtype=torch.float64, device=device)
        bb_mask = torch.from_numpy(cell_encodings_object_level_mask).to(device).transpose(1, 2)
        score_mask = torch.matmul(aa_mask, bb_mask)
        cc[score_mask == 0] = -100
        scores_2 = cc.max(dim=-1)[0].mean(dim=-1).cpu().numpy()
        S2[query_idx] = scores_2

        aa = torch.from_numpy(text_encodings_relation_level[query_idx]).to(device)
        bb = torch.from_numpy(cell_encodings_relation_level).to(device)
        aa = aa.unsqueeze(0).expand(n_cells, args.num_mentioned, args.coarse_embed_dim)
        bb = bb.transpose(1, 2)
        cc = torch.matmul(aa, bb)
        bb_mask = torch.from_numpy(cell_encodings_relation_level_mask).to(device).transpose(1, 2)
        score_mask = torch.matmul(aa_mask, bb_mask)
        cc[score_mask == 0] = -100
        scores_3 = cc.max(dim=-1)[0].mean(dim=-1).cpu().numpy()
        S3[query_idx] = scores_3

    return S1, S2, S3, query_cell_ids, db_cell_ids, args.top_k


if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    dataset_name = args.base_path[:-
                                  1] if args.base_path.endswith("/") else args.base_path
    dataset_name = dataset_name.split("/")[-1]
    print(f"Directory: {dataset_name}")

    cont = "Y" if bool(args.continue_path) else "N"
    feats = "all" if len(
        args.use_features) == 3 else "-".join(args.use_features)
    folder_name = args.folder_name
    print("#####################")
    print("########   Folder Name: " + folder_name)
    print("#####################")
    if not osp.isdir(f"./checkpoints/{dataset_name}/{folder_name}"):
        os.mkdir(f"./checkpoints/{dataset_name}/{folder_name}")

    """
    Create data loaders
    """
    if args.dataset == "K360":
        # ['2013_05_28_drive_0003_sync', ]
        if args.no_pc_augment:
            train_transform = T.FixedPoints(args.pointnet_numpoints)
            val_transform = T.FixedPoints(args.pointnet_numpoints)
        else:
            train_transform = T.Compose(
                [
                    T.FixedPoints(args.pointnet_numpoints),
                    T.RandomRotate(120, axis=2),
                    T.NormalizeScale(),
                ]
            )
            val_transform = T.Compose(
                [T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

        dataset_train = Kitti360CoarseDatasetMulti(
            args.base_path,
            SCENE_NAMES_TRAIN,
            train_transform,
            shuffle_hints=True,
            flip_poses=True,
        )

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=args.shuffle,
            num_workers=args.cpus,
        )

        dataset_val = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_VAL, val_transform,)

        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=False,
        )

        dataset_test = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_TEST, val_transform,)

        dataloader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=False,
        )

    assert sorted(dataset_train.get_known_classes()) == sorted(
        dataset_val.get_known_classes())

    data = dataset_train[0]
    assert len(data["debug_hint_descriptions"]) == args.num_mentioned
    batch = next(iter(dataloader_train))

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)

    lr = args.learning_rate

    dict_loss = {lr: []}
    dict_acc = {k: [] for k in args.top_k}
    dict_acc_val = {k: [] for k in args.top_k}
    dict_acc_val_close = {k: [] for k in args.top_k}
    dict_acc_test = {k: [] for k in args.top_k}
    dict_acc_test_close = {k: [] for k in args.top_k}

    best_val_accuracy = -1
    last_model_save_path_val = None
    last_optimizer_save_path_val = None
    model = CellRetrievalNetwork(
        dataset_train.get_known_classes(),
        COLOR_NAMES_K360,
        args,
    )
    if args.continue_path:
        model_dic = torch.load(
            args.continue_path, map_location=torch.device("cpu"))
        model.load_state_dict(model_dic, strict=False)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.ranking_loss == "pairwise":
        criterion = PairwiseRankingLoss(margin=args.margin)
    if args.ranking_loss == "hardest":
        criterion = HardestRankingLoss(margin=args.margin)
    if args.ranking_loss == "triplet":
        criterion = nn.TripletMarginLoss(margin=args.margin)
    if args.ranking_loss == "contrastive":
        criterion = ContrastiveLoss(temperature=args.temperature)
        criterion_2 = ContrastiveLoss_input_score(temperature=args.temperature)
    if args.ranking_loss == "CCL":
        criterion = CCL(temperature=args.temperature, alpha=args.alpha)
        criterion_2 = CCL_input_score(
            temperature=args.temperature, alpha=args.alpha)

    if args.lr_scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
    elif args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, args.lr_step, args.lr_gamma)
    else:
        raise TypeError

    for epoch in range(1, args.epochs + 1):
        # dataset_train.reset_seed() #OPTION: re-setting seed leads to equal data at every epoch
        loss, train_batches = train_epoch(model, dataloader_train, args)
        # train_acc, train_acc_close, train_retrievals = eval_epoch(
        #     model, dataloader_train, args
        # )
        val_acc, val_acc_close, val_retrievals = eval_epoch(
            model, dataloader_val, args)
        test_acc, test_acc_close, test_retrievals = eval_epoch(
            model, dataloader_test, args)

        key = lr
        dict_loss[key].append(loss)
        for k in args.top_k:
            # dict_acc[k].append(train_acc[k])
            dict_acc_val[k].append(val_acc[k])
            dict_acc_val_close[k].append(val_acc_close[k])
            dict_acc_test[k].append(test_acc[k])
            dict_acc_test_close[k].append(test_acc_close[k])

        scheduler.step()
        print(
            f"\t lr {lr:0.4} loss {loss:0.3f} epoch {epoch} train-acc: ", end="")
        # for k, v in train_acc.items():
        #     print(f"{k}-{v:0.3f} ", end="")
        print("val-acc: ", end="")
        for k, v in val_acc.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("val-acc-close: ", end="")
        for k, v in val_acc_close.items():
            print(f"{k}-{v:0.3f} ", end="")

        print("test-acc: ", end="")
        for k, v in test_acc.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("test-acc-close: ", end="")
        for k, v in test_acc_close.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("\n", flush=True)
        # save result to txt
        with open('output.txt', 'a+') as f:
            f.write(
                f"\t lr {lr:0.4} loss {loss:0.3f} epoch {epoch} train-acc: ")
            f.write("val-acc: ")
            for k, v in val_acc.items():
                f.write(f"{k}-{v:0.3f} ")
            f.write("val-acc-close: ")
            for k, v in val_acc_close.items():
                f.write(f"{k}-{v:0.3f} ")

            f.write("test-acc: ")
            for k, v in test_acc.items():
                f.write(f"{k}-{v:0.3f} ")
            f.write("test-acc-close: ")
            for k, v in test_acc_close.items():
                f.write(f"{k}-{v:0.3f} ")
            f.write("\n")
        # Saving best model
        acc_val = val_acc[max(args.top_k)]
        if acc_val > best_val_accuracy:
            model_path = f"./checkpoints/{dataset_name}/{folder_name}/coarse_cont{cont}_epoch{epoch}_acc{acc_val:0.3f}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_loss-{args.ranking_loss}_f-{feats}.pth"
            if not osp.isdir(osp.dirname(model_path)):
                os.mkdir(osp.dirname(model_path))

            print(f"Saving model at {acc_val:0.2f} to {model_path}")

            try:
                model_dic = model.state_dict()
                out = collections.OrderedDict()
                for item in model_dic:
                    if "llm_model" not in item:
                        out[item] = model_dic[item]
                torch.save(out, model_path)
                if (
                    last_model_save_path_val is not None
                    and last_model_save_path_val != model_path
                    and osp.isfile(last_model_save_path_val)
                ):
                    print("Removing", last_model_save_path_val)
                    os.remove(last_model_save_path_val)

                last_model_save_path_val = model_path

            except Exception as e:
                print(f"Error saving model!", str(e))
            best_val_accuracy = acc_val
