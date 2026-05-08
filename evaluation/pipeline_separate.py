"""
三分支融合评估脚本

加载三个独立分支的权重，分别计算相似度，然后相加得到最终结果。

使用方式:
    python evaluation/pipeline_separate.py \
        --global_checkpoint checkpoints/global/global_best.pth \
        --object_checkpoint checkpoints/object/object_best.pth \
        --relation_checkpoint checkpoints/relation/relation_best.pth

可选：只使用部分分支评估
    python evaluation/pipeline_separate.py \
        --object_checkpoint checkpoints/object/object_best.pth \
        --branches object
"""

import torch
import numpy as np
import os.path as osp
from easydict import EasyDict
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
import tqdm

from models.branches import GlobalBranch, ObjectBranch, RelationBranch
from evaluation.args import parse_arguments
from evaluation.utils import print_accuracies
from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from datapreparation.kitti360pose.utils import SCENE_NAMES_TEST, SCENE_NAMES_VAL, KNOWN_CLASS
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360


def z_score(tensor):
    """Z-score normalization"""
    mean = np.mean(tensor)
    std = np.std(tensor)
    if std == 0:
        return tensor
    return (tensor - mean) / std


@torch.no_grad()
def encode_all_branches(global_model, object_model, relation_model, dataloader, cells_dataloader, args):
    """
    使用三个分支分别编码查询和场景
    
    Returns:
        - text_enc_global: [N_queries, embed_dim]
        - cell_enc_global: [N_cells, embed_dim]
        - text_enc_object: [N_queries, num_mentioned, embed_dim]
        - cell_enc_object: [N_cells, object_size, embed_dim]
        - cell_mask_object: [N_cells, object_size, 1]
        - text_enc_relation: [N_queries, num_mentioned, embed_dim]
        - cell_enc_relation: [N_cells, object_size, embed_dim]
        - cell_mask_relation: [N_cells, object_size, 1]
        - query_cell_ids: [N_queries]
        - db_cell_ids: [N_cells]
    """
    # Initialize output arrays
    n_queries = len(dataloader.dataset)
    n_cells = len(cells_dataloader.dataset)
    embed_dim = args.coarse_embed_dim
    
    # Global encodings
    text_enc_global = np.zeros((n_queries, embed_dim), dtype=np.float32)
    cell_enc_global = np.zeros((n_cells, embed_dim), dtype=np.float32)
    
    # Object encodings
    text_enc_object = np.zeros((n_queries, args.num_mentioned, embed_dim), dtype=np.float32)
    cell_enc_object = np.zeros((n_cells, args.object_size, embed_dim), dtype=np.float32)
    cell_mask_object = np.zeros((n_cells, args.object_size, 1), dtype=np.float32)
    
    # Relation encodings
    text_enc_relation = np.zeros((n_queries, args.num_mentioned, embed_dim), dtype=np.float32)
    cell_enc_relation = np.zeros((n_cells, args.object_size, embed_dim), dtype=np.float32)
    cell_mask_relation = np.zeros((n_cells, args.object_size, 1), dtype=np.float32)
    
    query_cell_ids = []
    db_cell_ids = []

    # Encode queries
    print("\n  Encoding queries...")
    index_offset = 0
    for batch in tqdm.tqdm(dataloader, desc="  Query encoding"):
        batch_size = len(batch["texts"])
        
        # Global branch
        anchor = global_model.encode_text(batch["texts"])
        text_enc_global[index_offset:index_offset + batch_size] = anchor.cpu().numpy()
        
        # Object branch
        F_Object_T = object_model.encode_text(batch["texts"])
        text_enc_object[index_offset:index_offset + batch_size] = F_Object_T.cpu().numpy()
        
        # Relation branch
        F_Relation_T = relation_model.encode_text(batch["texts"])
        text_enc_relation[index_offset:index_offset + batch_size] = F_Relation_T.cpu().numpy()
        
        query_cell_ids.extend(batch["cell_ids"])
        index_offset += batch_size

    # Encode cells
    print("\n  Encoding cells...")
    index_offset = 0
    for batch in tqdm.tqdm(cells_dataloader, desc="  Cell encoding"):
        batch_size = len(batch["cell_ids"])
        
        # Global branch
        F_Global = global_model.encode_objects(batch["objects"], batch["object_points"])
        cell_enc_global[index_offset:index_offset + batch_size] = F_Global.cpu().numpy()
        
        # Object branch
        F_Object_P, mask_object = object_model.encode_objects(batch["objects"], batch["object_points"])
        cell_enc_object[index_offset:index_offset + batch_size] = F_Object_P.cpu().numpy()
        cell_mask_object[index_offset:index_offset + batch_size] = mask_object.cpu().numpy()
        
        # Relation branch
        F_Relation_P, mask_relation = relation_model.encode_objects(batch["objects"], batch["object_points"])
        cell_enc_relation[index_offset:index_offset + batch_size] = F_Relation_P.cpu().numpy()
        cell_mask_relation[index_offset:index_offset + batch_size] = mask_relation.cpu().numpy()
        
        db_cell_ids.extend(batch["cell_ids"])
        index_offset += batch_size

    return {
        'text_enc_global': text_enc_global,
        'cell_enc_global': cell_enc_global,
        'text_enc_object': text_enc_object,
        'cell_enc_object': cell_enc_object,
        'cell_mask_object': cell_mask_object,
        'text_enc_relation': text_enc_relation,
        'cell_enc_relation': cell_enc_relation,
        'cell_mask_relation': cell_mask_relation,
        'query_cell_ids': np.array(query_cell_ids),
        'db_cell_ids': np.array(db_cell_ids),
    }


def compute_scores(encodings, args, branches=['global', 'object', 'relation'], weights=None):
    """
    计算三分支的相似度分数
    
    Args:
        encodings: encode_all_branches 的返回值
        branches: 要使用的分支列表
        weights: 权重字典 {'global': 1.0, 'object': 1.0, 'relation': 1.0}
    
    Returns:
        scores: 最终相似度分数 [N_queries, N_cells]
        scores_breakdown: 各分支的分数字典
    """
    if weights is None:
        weights = {
            'global': getattr(args, 'weight_global', 1.0),
            'object': getattr(args, 'weight_object', 1.0),
            'relation': getattr(args, 'weight_relation', 1.0),
        }
    
    scores_breakdown = {}
    final_scores = None
    
    # Global scores
    if 'global' in branches:
        scores_global = encodings['text_enc_global'] @ encodings['cell_enc_global'].T
        scores_breakdown['global'] = scores_global
        weighted_score = weights['global'] * scores_global
        if final_scores is None:
            final_scores = weighted_score.copy()
        else:
            final_scores += weighted_score
        print(f"    Global Score Range: [{scores_global.min():.3f}, {scores_global.max():.3f}], weight: {weights['global']:.2f}")
    
    # Object scores
    if 'object' in branches:
        scores_object = np.zeros((len(encodings['query_cell_ids']), len(encodings['db_cell_ids'])))
        for query_idx in range(len(encodings['query_cell_ids'])):
            aa = torch.from_numpy(encodings['text_enc_object'][query_idx])
            bb = torch.from_numpy(encodings['cell_enc_object'])
            aa = aa.unsqueeze(0).expand(len(encodings['cell_enc_object']), args.num_mentioned, args.coarse_embed_dim)
            bb = bb.transpose(1, 2)
            cc = torch.matmul(aa, bb)
            aa_mask = torch.ones((args.num_mentioned, 1))
            bb_mask = torch.from_numpy(encodings['cell_mask_object']).transpose(1, 2)
            score_mask = torch.matmul(aa_mask, bb_mask)
            cc[score_mask == 0] = -100
            scores_object[query_idx] = cc.max(dim=-1)[0].mean(dim=-1).numpy()
        scores_breakdown['object'] = scores_object
        weighted_score = weights['object'] * scores_object
        if final_scores is None:
            final_scores = weighted_score.copy()
        else:
            final_scores += weighted_score
        print(f"    Object Score Range: [{scores_object.min():.3f}, {scores_object.max():.3f}], weight: {weights['object']:.2f}")
    
    # Relation scores
    if 'relation' in branches:
        scores_relation = np.zeros((len(encodings['query_cell_ids']), len(encodings['db_cell_ids'])))
        for query_idx in range(len(encodings['query_cell_ids'])):
            aa = torch.from_numpy(encodings['text_enc_relation'][query_idx])
            bb = torch.from_numpy(encodings['cell_enc_relation'])
            aa = aa.unsqueeze(0).expand(len(encodings['cell_enc_relation']), args.num_mentioned, args.coarse_embed_dim)
            bb = bb.transpose(1, 2)
            cc = torch.matmul(aa, bb)
            aa_mask = torch.ones((args.num_mentioned, 1))
            bb_mask = torch.from_numpy(encodings['cell_mask_relation']).transpose(1, 2)
            score_mask = torch.matmul(aa_mask, bb_mask)
            cc[score_mask == 0] = -100
            scores_relation[query_idx] = cc.max(dim=-1)[0].mean(dim=-1).numpy()
        scores_breakdown['relation'] = scores_relation
        weighted_score = weights['relation'] * scores_relation
        if final_scores is None:
            final_scores = weighted_score.copy()
        else:
            final_scores += weighted_score
        print(f"    Relation Score Range: [{scores_relation.min():.3f}, {scores_relation.max():.3f}], weight: {weights['relation']:.2f}")
    
    print(f"    Final Score Range: [{final_scores.min():.3f}, {final_scores.max():.3f}]")
    
    return final_scores, scores_breakdown


def evaluate(scores, encodings, args):
    """
    评估检索准确率
    
    Args:
        scores: [N_queries, N_cells] 相似度分数
        encodings: 包含 query_cell_ids 和 db_cell_ids
        args: 评估参数
    
    Returns:
        accuracies: Hit@K 准确率
        accuracies_close: Close@K 准确率
    """
    query_cell_ids = encodings['query_cell_ids']
    db_cell_ids = encodings['db_cell_ids']
    top_k = args.top_k
    
    # Get cell dataset for distance calculation
    cells_dataset = encodings.get('cells_dataset')
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size
    
    # Get query poses
    query_poses_w = encodings.get('query_poses_w')
    
    accuracies = {k: [] for k in top_k}
    accuracies_close = {k: [] for k in top_k}
    
    for query_idx in range(len(query_cell_ids)):
        sorted_indices = np.argsort(-scores[query_idx])
        target_cell_id = query_cell_ids[query_idx]
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        
        # Hit@K
        for k in top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[:k])
        
        # Close@K
        if query_poses_w is not None:
            target_pose_w = query_poses_w[query_idx]
            retrieved_cell_poses = [
                cells_dict[cell_id].get_center()[0:2] for cell_id in retrieved_cell_ids
            ]
            dists = np.linalg.norm(target_pose_w - retrieved_cell_poses, axis=1)
            for k in top_k:
                accuracies_close[k].append(np.any(dists[:k] <= cell_size / 2))
    
    for k in top_k:
        accuracies[k] = np.mean(accuracies[k])
        accuracies_close[k] = np.mean(accuracies_close[k])
    
    return accuracies, accuracies_close


def add_separate_evaluation_args(parser):
    """添加三分支评估相关的命令行参数"""
    parser.add_argument('--global_checkpoint', type=str, default=None,
                        help='Path to Global branch checkpoint')
    parser.add_argument('--object_checkpoint', type=str, default=None,
                        help='Path to Object branch checkpoint')
    parser.add_argument('--relation_checkpoint', type=str, default=None,
                        help='Path to Relation branch checkpoint')
    parser.add_argument('--branches', type=str, nargs='+',
                        default=['global', 'object', 'relation'],
                        choices=['global', 'object', 'relation'],
                        help='Which branches to use for evaluation (default: all)')
    # 权重参数（用于加权融合）
    parser.add_argument('--weight_global', type=float, default=1.0,
                        help='Weight for Global branch (default: 1.0)')
    parser.add_argument('--weight_object', type=float, default=1.0,
                        help='Weight for Object branch (default: 1.0)')
    parser.add_argument('--weight_relation', type=float, default=1.0,
                        help='Weight for Relation branch (default: 1.0)')


def main():
    import argparse
    # 使用create_base_parser而不是parse_arguments，避免参数被提前解析
    from evaluation.args import create_base_parser
    parser = create_base_parser()
    add_separate_evaluation_args(parser)
    
    args = parser.parse_args()
    args = EasyDict(vars(args))
    
    # Validate branches
    if not args.branches:
        print("Error: No branches specified for evaluation!")
        return
    
    print(f"\n{'#'*70}")
    print(f"# Separate Branch Evaluation")
    print(f"{'#'*70}")
    print(f"# Branches: {args.branches}")
    print(f"# Checkpoints:")
    if 'global' in args.branches and args.global_checkpoint:
        print(f"#   - Global: {args.global_checkpoint}")
    if 'object' in args.branches and args.object_checkpoint:
        print(f"#   - Object: {args.object_checkpoint}")
    if 'relation' in args.branches and args.relation_checkpoint:
        print(f"#   - Relation: {args.relation_checkpoint}")
    print(f"{'#'*70}\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}\n")

    # Create transforms
    if args.no_pc_augment:
        transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform = T.Compose([
            T.FixedPoints(args.pointnet_numpoints),
            T.NormalizeScale()
        ])

    # Create datasets
    if args.use_test_set:
        dataset = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_TEST, transform,
            shuffle_hints=False, flip_poses=False,
        )
    else:
        dataset = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_VAL, transform,
            shuffle_hints=False, flip_poses=False,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    cells_dataset = dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    # Load models
    print("Loading models...")
    models = {}
    
    # Global model
    if 'global' in args.branches and args.global_checkpoint:
        model_global = GlobalBranch(KNOWN_CLASS, COLOR_NAMES_K360, args)
        model_global.load_state_dict(torch.load(args.global_checkpoint, map_location='cpu'), strict=False)
        model_global.to(device)
        model_global.eval()
        models['global'] = model_global
        print(f"  ✓ Global: {args.global_checkpoint}")
    else:
        models['global'] = None
        
    # Object model
    if 'object' in args.branches and args.object_checkpoint:
        model_object = ObjectBranch(KNOWN_CLASS, COLOR_NAMES_K360, args)
        model_object.load_state_dict(torch.load(args.object_checkpoint, map_location='cpu'), strict=False)
        model_object.to(device)
        model_object.eval()
        models['object'] = model_object
        print(f"  ✓ Object: {args.object_checkpoint}")
    else:
        models['object'] = None
        
    # Relation model
    if 'relation' in args.branches and args.relation_checkpoint:
        model_relation = RelationBranch(KNOWN_CLASS, COLOR_NAMES_K360, args)
        model_relation.load_state_dict(torch.load(args.relation_checkpoint, map_location='cpu'), strict=False)
        model_relation.to(device)
        model_relation.eval()
        models['relation'] = model_relation
        print(f"  ✓ Relation: {args.relation_checkpoint}")
    else:
        models['relation'] = None

    # Check if at least one model is loaded
    if not any(models.values()):
        print("Error: No checkpoints loaded!")
        return

    # Get query poses for close accuracy calculation
    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataset.all_poses])

    # Encode all branches
    encodings = encode_all_branches(
        models['global'], models['object'], models['relation'],
        dataloader, cells_dataloader, args
    )
    encodings['cells_dataset'] = cells_dataset
    encodings['query_poses_w'] = query_poses_w

    # Compute scores
    print("\n  Computing similarities...")
    scores, scores_breakdown = compute_scores(encodings, args, args.branches)

    # Evaluate
    print("\n  Evaluating...")
    accuracies, accuracies_close = evaluate(scores, encodings, args)

    # Print results
    weights_used = {
        'global': getattr(args, 'weight_global', 1.0),
        'object': getattr(args, 'weight_object', 1.0),
        'relation': getattr(args, 'weight_relation', 1.0),
    }
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Evaluation Results                                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Branches used: {', '.join(args.branches)}                                                   ║
║  Weights: Global={weights_used['global']:.2f}, Object={weights_used['object']:.2f}, Relation={weights_used['relation']:.2f}            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Hit@K Accuracy:                                                              ║""")
    for k in args.top_k:
        print(f"║    Hit@{k:2d}: {accuracies[k]:.4f}                                                ║")
    print(f"""╠══════════════════════════════════════════════════════════════════════╣
║  Close@K Accuracy:                                                    ║""")
    for k in args.top_k:
        print(f"║    Close@{k:2d}: {accuracies_close[k]:.4f}                                              ║")
    print(f"""╚══════════════════════════════════════════════════════════════════════╝""")

    # Print individual branch results for comparison
    if len(args.branches) > 1:
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  Individual Branch Results (for comparison)                           ║""")
        for branch in args.branches:
            if branch in scores_breakdown:
                branch_encodings = {
                    'query_cell_ids': encodings['query_cell_ids'],
                    'db_cell_ids': encodings['db_cell_ids'],
                    'cells_dataset': cells_dataset,
                    'query_poses_w': query_poses_w,
                }
                acc, acc_close = evaluate(scores_breakdown[branch], branch_encodings, args)
                print(f"╠══════════════════════════════════════════════════════════════════════╣")
                print(f"║  {branch.upper()} Branch:                                                      ║")
                print(f"║    Hit@{max(args.top_k):2d}: {acc[max(args.top_k)]:.4f}   Close@{max(args.top_k):2d}: {acc_close[max(args.top_k)]:.4f}                        ║")
        print(f"""╚══════════════════════════════════════════════════════════════════════╝""")


if __name__ == "__main__":
    main()
