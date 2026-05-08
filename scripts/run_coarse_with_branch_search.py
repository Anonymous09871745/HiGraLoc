#!/usr/bin/env python3
"""
与 evaluation.coarse 完全相同的评估流程，并额外在验证集上对三路分支权重 (w1, w2, w3) 做网格搜索。

用法与 evaluation.coarse 一致，例如:
  python scripts/run_coarse_with_branch_search.py \\
    --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \\
    --use_features class color position num \\
    --use_test_set \\
    --no_pc_augment \\
    --hungging_model t5-large \\
    --fixed_embedding \\
    --path_coarse ./checkpoints/.../coarse_contN_epoch23_....pth

输出: 先与 evaluation.coarse 相同（Encoded ... / Retrieval Accs / Retrieval Accs Close / Coarse 表），
      再输出在验证集上的分支权重网格搜索结果。
"""

import sys
import os

# 过滤空参数，避免复制粘贴换行导致 unrecognized arguments
sys.argv = [a for a in sys.argv if a.strip()]

# 项目根
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch_geometric.transforms as T

from evaluation.args import parse_arguments
from evaluation.coarse import run_coarse
from evaluation.utils import print_accuracies
from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from datapreparation.kitti360pose.utils import SCENE_NAMES_TEST, SCENE_NAMES_VAL, KNOWN_CLASS
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from models.cell_retrieval import CellRetrievalNetwork
from training.coarse import eval_epoch_return_score_matrices

# 网格搜索默认范围
GRID = [0.5, 0.75, 1.0, 1.25, 1.5]


def z_score_per_row(arr, eps=1e-8):
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (arr - mean) / std


def eval_weights(S1, S2, S3, query_cell_ids, db_cell_ids, w1, w2, w3, top_k_list):
    z1 = z_score_per_row(S1)
    z2 = z_score_per_row(S2)
    z3 = z_score_per_row(S3)
    fused = w1 * z1 + w2 * z2 + w3 * z3
    n_queries = fused.shape[0]
    max_k = max(top_k_list)
    acc = {k: 0.0 for k in top_k_list}
    for q in range(n_queries):
        order = np.argsort(-fused[q])[:max_k]
        retrieved = db_cell_ids[order]
        target = query_cell_ids[q]
        for k in top_k_list:
            if target in retrieved[:k]:
                acc[k] += 1.0
    for k in top_k_list:
        acc[k] /= n_queries
    return acc


def main():
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))

    if args.no_pc_augment:
        transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    if args.use_test_set:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_TEST, transform, shuffle_hints=False, flip_poses=False,
        )
    else:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_VAL, transform, shuffle_hints=False, flip_poses=False,
        )

    dataloader_retrieval = DataLoader(
        dataset_retrieval,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    model_coarse_dic = torch.load(args.path_coarse, map_location=torch.device("cpu"))
    model_coarse = CellRetrievalNetwork(KNOWN_CLASS, COLOR_NAMES_K360, args)
    model_coarse.load_state_dict(model_coarse_dic, strict=False)
    model_coarse.to(device)

    # 与 evaluation.coarse 完全一致：跑检索并打印
    retrievals, coarse_accuracies = run_coarse(model_coarse, dataloader_retrieval, args)
    print_accuracies(coarse_accuracies, "Coarse")

    # 在验证集上做分支权重网格搜索（与主流程使用的 test/val 无关，固定用 val）
    print("\n" + "=" * 60)
    print("Branch weight grid search on validation set")
    print("=" * 60)

    dataset_val = Kitti360CoarseDatasetMulti(
        args.base_path, SCENE_NAMES_VAL, transform, shuffle_hints=False, flip_poses=False,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    S1, S2, S3, query_cell_ids, db_cell_ids, top_k_list = eval_epoch_return_score_matrices(
        model_coarse, dataloader_val, args
    )

    acc_default = eval_weights(S1, S2, S3, query_cell_ids, db_cell_ids, 1.0, 1.0, 1.0, top_k_list)
    print("Default (1.0, 1.0, 1.0) on val:")
    print(acc_default)

    best_acc = acc_default
    best_weights = (1.0, 1.0, 1.0)
    for w1 in GRID:
        for w2 in GRID:
            for w3 in GRID:
                acc = eval_weights(S1, S2, S3, query_cell_ids, db_cell_ids, w1, w2, w3, top_k_list)
                if acc[1] > best_acc[1]:
                    best_acc = acc
                    best_weights = (w1, w2, w3)

    print("Best (w1, w2, w3) by val acc@1:", best_weights)
    print("Best val accs:", best_acc)


if __name__ == "__main__":
    main()
