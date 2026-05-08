#!/usr/bin/env python3
"""
在验证集上搜索 coarse 三路融合的最优分支权重 (w1, w2, w3)。

用法（在项目根目录 MNCL 下执行）:
  python scripts/search_branch_weights.py \
    --checkpoint checkpoints/k360_30-10_scG_pd10_pc4_spY_all/exp_coarse_staged/coarse_contN_epoch26_acc0.874_ecl0_eco0_p256_npa1_loss-CCL_f-class-color-position-num.pth \
    --base_path /path/to/Kitti360Pose \
    --hungging_model t5-large \
    --out results_branch_weights.txt

  可选: --grid 0.5 1.0 1.5, --metric top1|top3|top5
  离线/无外网: 加 --offline，且 --hungging_model 与训练一致（如 t5-large）以使用本地缓存

融合分: score = w1 * z_score(scores_global) + w2 * z_score(scores_object) + w3 * z_score(scores_relation)
目标: 在验证集上使 top-1 / top-3 等准确度最高的 (w1, w2, w3)。
"""

import argparse
import os
import sys

# 离线模式：在导入 transformers 前设置，避免连接 huggingface.co 超时
if "--offline" in sys.argv:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
import tqdm

# 项目根目录加入 path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.cell_retrieval import CellRetrievalNetwork
from datapreparation.kitti360pose.utils import SCENE_NAMES_VAL, COLOR_NAMES as COLOR_NAMES_K360, KNOWN_CLASS
from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti, Kitti360CoarseDataset
from training.coarse import eval_epoch_return_score_matrices


def z_score_per_row(arr, eps=1e-8):
    """按行 z-score，arr shape (n_rows, n_cols)。std 用 eps 保底避免除零。"""
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    std = np.maximum(std, eps)
    return (arr - mean) / std


def build_args(checkpoint_path, base_path, hungging_model, **kwargs):
    """构建与 checkpoint 一致的 args（可从 checkpoint 文件名推断部分参数）。"""
    from argparse import Namespace
    # 从 checkpoint 文件名推断: p256, npa1, loss-CCL, f-class-color-position-num
    default = Namespace(
        batch_size=kwargs.get("batch_size", 32),
        dataset="K360",
        base_path=base_path,
        use_features=["class", "color", "position", "num"],
        coarse_embed_dim=256,
        pointnet_numpoints=256,
        pointnet_layers=3,
        pointnet_variation=0,
        pointnet_path=kwargs.get("pointnet_path", "./checkpoints/pointnet_acc0.86_lr1_p256.pth"),
        pointnet_freeze=False,
        pointnet_features=2,
        no_pc_augment=True,  # npa1
        object_size=28,
        object_inter_module_num_heads=4,
        object_inter_module_num_layers=2,
        class_embed=False,
        color_embed=False,
        num_mentioned=6,
        hungging_model=hungging_model,
        fixed_embedding=kwargs.get("fixed_embedding", True),
        inter_module_num_heads=4,
        inter_module_num_layers=1,
        intra_module_num_heads=4,
        intra_module_num_layers=1,
        num_of_hidden_layer=3,
        top_k=kwargs.get("top_k", [1, 3, 5]),
        ranking_loss="CCL",
        temperature=0.1,
        alpha=2,
        cpus=0,
    )
    return default


def load_model_and_encode(args, checkpoint_path, device):
    """与 evaluation.coarse / training.coarse.eval_epoch 完全一致：用 eval_epoch_return_score_matrices 得到 S1,S2,S3。"""
    transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])
    dataset_val = Kitti360CoarseDatasetMulti(
        args.base_path, SCENE_NAMES_VAL, transform, shuffle_hints=False, flip_poses=False,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
        num_workers=args.cpus,
    )
    model = CellRetrievalNetwork(KNOWN_CLASS, COLOR_NAMES_K360, args)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = {k: v for k, v in ckpt.items() if "llm_model" not in k}
    model.load_state_dict(state, strict=False)
    model.to(device)
    S1, S2, S3, query_cell_ids, db_cell_ids, top_k_list = eval_epoch_return_score_matrices(
        model, dataloader_val, args
    )
    return S1, S2, S3, query_cell_ids, db_cell_ids, top_k_list


def evaluate_weights(S1, S2, S3, query_cell_ids, db_cell_ids, w1, w2, w3, top_k_list, z_eps=1e-8):
    """给定 (w1,w2,w3)，计算融合分并返回各 top_k 准确率。"""
    z1 = z_score_per_row(S1, eps=z_eps)
    z2 = z_score_per_row(S2, eps=z_eps)
    z3 = z_score_per_row(S3, eps=z_eps)
    fused = w1 * z1 + w2 * z2 + w3 * z3  # (n_queries, n_cells)
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
    parser = argparse.ArgumentParser(description="Search branch weights (w1,w2,w3) on val set.")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/k360_30-10_scG_pd10_pc4_spY_all/exp_coarse_staged/coarse_contN_epoch26_acc0.874_ecl0_eco0_p256_npa1_loss-CCL_f-class-color-position-num.pth",
                        help="Path to coarse checkpoint (relative to project root).")
    parser.add_argument("--base_path", type=str, required=True, help="Kitti360Pose root path.")
    parser.add_argument("--hungging_model", type=str, default="t5-large",
                        help="HuggingFace model name or local path (须与训练时一致，如 t5-large).")
    parser.add_argument("--offline", action="store_true",
                        help="Use TRANSFORMERS_OFFLINE=1 to avoid connecting to HuggingFace (use cached model).")
    parser.add_argument("--grid", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5],
                        help="Candidate values for each weight in [0.5, 1.5] (same for w1,w2,w3).")
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5], help="top-k for accuracy.")
    parser.add_argument("--metric", type=str, default="top1", choices=["top1", "top3", "top5"],
                        help="Which metric to maximize for best (w1,w2,w3).")
    parser.add_argument("--out", type=str, default=None, help="Save best (w1,w2,w3) and results to file.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pointnet_path", type=str, default="./checkpoints/pointnet_acc0.86_lr1_p256.pth")
    parser.add_argument("--fixed_embedding", action="store_true", default=True,
                        help="Match evaluation.coarse (default True).")
    parser.add_argument("--no_fixed_embedding", action="store_true", help="Disable fixed_embedding.")
    # 过滤掉空参数（复制粘贴或换行产生的空字符串）
    sys.argv = [a for a in sys.argv if a.strip()]
    args = parser.parse_args()
    if args.no_fixed_embedding:
        args.fixed_embedding = False

    # checkpoint 若为相对路径则相对于项目根
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(ROOT, args.checkpoint)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Base path:", args.base_path)
    print("Grid:", args.grid)
    print("Metric to maximize:", args.metric)

    run_args = build_args(
        args.checkpoint, args.base_path, args.hungging_model,
        batch_size=args.batch_size, top_k=args.top_k, pointnet_path=args.pointnet_path,
        fixed_embedding=args.fixed_embedding,
    )

    print("Loading model and encoding val set...")
    S1, S2, S3, query_cell_ids, db_cell_ids, top_k_list = load_model_and_encode(
        run_args, args.checkpoint, device
    )
    n_queries, n_cells = S1.shape
    print(f"Queries: {n_queries}, Cells: {n_cells}")

    # 默认 (1,1,1) 的准确率
    acc_default = evaluate_weights(S1, S2, S3, query_cell_ids, db_cell_ids, 1.0, 1.0, 1.0, top_k_list)
    print("\nDefault (1.0, 1.0, 1.0):")
    for k in top_k_list:
        print(f"  acc@{k}: {acc_default[k]:.4f}")

    k_to_idx = {"top1": 1, "top3": 3, "top5": 5}
    best_metric_key = k_to_idx[args.metric]
    if best_metric_key not in top_k_list:
        top_k_list = sorted(set(top_k_list) | {best_metric_key})
    best_val = -1.0
    best_weights = (1.0, 1.0, 1.0)
    best_acc = None
    results = []

    for w1 in args.grid:
        for w2 in args.grid:
            for w3 in args.grid:
                acc = evaluate_weights(S1, S2, S3, query_cell_ids, db_cell_ids, w1, w2, w3, top_k_list)
                metric_val = acc[best_metric_key]
                results.append((w1, w2, w3, acc, metric_val))
                if metric_val > best_val:
                    best_val = metric_val
                    best_weights = (w1, w2, w3)
                    best_acc = acc

    print(f"\nBest (w1, w2, w3) by {args.metric}: {best_weights}")
    print("Best accuracies:")
    for k in sorted(best_acc.keys()):
        print(f"  acc@{k}: {best_acc[k]:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            f.write(f"checkpoint: {args.checkpoint}\n")
            f.write(f"base_path: {args.base_path}\n")
            f.write(f"best_weights (w1,w2,w3): {best_weights}\n")
            f.write(f"best_acc: {best_acc}\n")
            f.write("\nAll grid results (w1, w2, w3, acc per k, metric):\n")
            for w1, w2, w3, acc, mv in sorted(results, key=lambda x: -x[4]):
                acc_str = " ".join(f"@{k}={acc[k]:.4f}" for k in sorted(acc.keys()))
                f.write(f"  ({w1},{w2},{w3}) -> {acc_str} (metric={mv:.4f})\n")
        print("Results written to", args.out)


if __name__ == "__main__":
    main()
