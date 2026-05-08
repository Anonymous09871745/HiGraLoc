"""
联合评估脚本：三分支 Coarse + Fine 阶段

使用三分支权重搜索的最佳权重做 Coarse 阶段，然后使用 Fine 模型做精细定位。

使用方式:
    cd /root/autodl-tmp/MNCL

    python -m evaluation.pipeline_three_branch_with_fine \
        --global_checkpoint ./checkpoints/global/global_epoch21_acc0.7841.pth \
        --object_checkpoint ./checkpoints/object/object_epoch15_acc0.8754.pth \
        --relation_checkpoint ./checkpoints/relation/relation_epoch14_acc0.8701.pth \
        --path_fine ./checkpoints/k360_30-10_scG_pd10_pc4_spY_all/path_to_fine/fine_contN_epoch26_offset0.093_lr0.0003_obj-6-16_ecl0_eco0_p256_npa1_f-class-color-position-num.pth \
        --weight_global 0.5 \
        --weight_object 1.5 \
        --weight_relation 0.5 \
        --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
        --use_test_set \
        --batch_size 32 \
        --coarse_embed_dim 256 \
        --no_pc_augment \
        --fixed_embedding \
        --use_features class color position num \
        --hugging_model t5-large \
        --pointnet_path ./checkpoints/pointnet_acc0.86_lr1_p256.pth \
        --num_mentioned 6 \
        --object_size 28 \
        --inter_module_num_heads 4 \
        --inter_module_num_layers 1 \
        --intra_module_num_heads 4 \
        --intra_module_num_layers 1 \
        --num_of_hidden_layer 3 \
        --alpha 2

权重配置（最佳权重，来自参数搜索）:
    --weight_global 0.5
    --weight_object 1.5
    --weight_relation 0.5
"""

import torch
import numpy as np
import os
import os.path as osp
from easydict import EasyDict
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
import tqdm
from copy import deepcopy
import logging
import sys
from datetime import datetime

from models.branches import GlobalBranch, ObjectBranch, RelationBranch
from models.cross_matcher import CrossMatch
from evaluation.args import create_base_parser
from evaluation.utils import calc_sample_accuracies, print_accuracies
from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from dataloading.kitti360pose.eval import Kitti360TopKDataset
from datapreparation.kitti360pose.utils import SCENE_NAMES_TEST, SCENE_NAMES_VAL, KNOWN_CLASS
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from datapreparation.kitti360pose.imports import Object3d


def setup_logging(args):
    """设置日志系统，同时输出到文件和控制台"""
    # 创建日志目录
    log_dir = osp.join(osp.dirname(args.base_path.rstrip('/')), 'evaluation_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名，包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_names = f"{osp.basename(args.global_checkpoint)}_{osp.basename(args.object_checkpoint)}_{osp.basename(args.relation_checkpoint)}"
    checkpoint_names = checkpoint_names[:50]  # 限制长度
    log_filename = f"eval_{timestamp}_{checkpoint_names}.log"
    log_path = osp.join(log_dir, log_filename)
    
    # 创建 logger
    logger = logging.getLogger('ThreeBranchFineEval')
    logger.setLevel(logging.DEBUG)
    
    # 避免重复添加 handler
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件 handler - 详细日志
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台 handler - info 及以上
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件路径: {log_path}")
    logger.info(f"日志保存目录: {log_dir}")
    
    return logger, log_path


def add_three_branch_args(parser):
    """添加三分支评估相关的命令行参数"""
    parser.add_argument('--global_checkpoint', type=str, required=True,
                        help='Path to Global branch checkpoint')
    parser.add_argument('--object_checkpoint', type=str, required=True,
                        help='Path to Object branch checkpoint')
    parser.add_argument('--relation_checkpoint', type=str, required=True,
                        help='Path to Relation branch checkpoint')
    # Note: --path_fine is already defined in create_base_parser()
    # 三分支权重（最佳权重）
    parser.add_argument('--weight_global', type=float, default=0.5,
                        help='Weight for Global branch (default: 0.5)')
    parser.add_argument('--weight_object', type=float, default=1.5,
                        help='Weight for Object branch (default: 1.5)')
    parser.add_argument('--weight_relation', type=float, default=0.5,
                        help='Weight for Relation branch (default: 0.5)')
    parser.add_argument('--save_retrieval_json', type=str, default=None,
                        help='Path to save visualization-ready retrieval json')
    parser.add_argument('--vis_output_dir', type=str, default=None,
                        help='Optional directory to render visualization png files')
    parser.add_argument('--vis_topk', type=int, default=5,
                        help='Number of candidates to visualize')
    parser.add_argument('--vis_num_queries', type=int, default=100,
                        help='Number of queries to export for visualization')


def _cell_center_xy(cell):
    center = cell.get_center()
    return float(center[0]), float(center[1])


def build_viz_records(all_poses, all_cells, retrievals, combined_scores, offsets, args):
    records = []
    cell_dict = {cell.id: cell for cell in all_cells}
    max_queries = min(len(all_poses), int(args.vis_num_queries))

    for q_idx, pose in enumerate(all_poses[:max_queries]):
        top_cell_ids = list(retrievals[q_idx])
        candidates = []
        for rank, cell_id in enumerate(top_cell_ids[: args.vis_topk]):
            cell = cell_dict[cell_id]
            fine_xy = None
            if offsets is not None:
                if q_idx < len(offsets) and rank < len(offsets[q_idx]):
                    off = np.asarray(offsets[q_idx][rank])
                    center = np.asarray(cell.get_center()[0:2], dtype=np.float32)
                    fine_xy = tuple((center + off[0:2]).tolist())
            candidates.append({
                'cell_id': cell_id,
                'center_xy': _cell_center_xy(cell),
                'coarse_score': float(combined_scores[q_idx, np.where(np.array([c.id for c in all_cells]) == cell_id)[0][0]]),
                'fine_xy': fine_xy,
                'fine_score': None,
            })
        records.append({
            'query_id': pose.cell_id if hasattr(pose, 'cell_id') else str(q_idx),
            'query_text': pose.get_text(),
            'gt_xy': (float(pose.pose_w[0]), float(pose.pose_w[1])),
            'query_points_xy': [],
            'candidates': candidates,
            'metadata': {
                'scene_name': pose.scene_name,
                'topk': args.vis_topk,
            },
        })
    return records


@torch.no_grad()
def encode_all_branches(global_model, object_model, relation_model, dataloader, cells_dataloader, args, logger):
    """
    使用三个分支分别编码查询和场景
    """
    logger.info("=" * 80)
    logger.info("[Step 1/4] 开始编码查询和场景")
    logger.info("=" * 80)
    
    n_queries = len(dataloader.dataset)
    n_cells = len(cells_dataloader.dataset)
    embed_dim = args.coarse_embed_dim
    logger.info(f"配置参数: embed_dim={embed_dim}, num_mentioned={args.num_mentioned}, object_size={args.object_size}")

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
    logger.info("-" * 60)
    logger.info("Phase A: 开始编码查询 (Queries)")
    logger.info(f"  - 查询总数: {n_queries}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - 预期 batch 数量: {(n_queries + args.batch_size - 1) // args.batch_size}")
    logger.info("-" * 60)
    
    logger.info("  [Global branch] 开始编码文本查询...")
    logger.info("  [Object branch] 开始编码文本查询...")
    logger.info("  [Relation branch] 开始编码文本查询...")
    
    index_offset = 0
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="  Query encoding", leave=False)):
        batch_size = len(batch["texts"])
        logger.debug(f"  Batch {batch_idx}: 处理 {batch_size} 个查询")

        # Global branch
        anchor = global_model.encode_text(batch["texts"])
        text_enc_global[index_offset:index_offset + batch_size] = anchor.cpu().numpy()
        logger.debug(f"    Global encoding shape: {anchor.shape}")

        # Object branch
        F_Object_T = object_model.encode_text(batch["texts"])
        text_enc_object[index_offset:index_offset + batch_size] = F_Object_T.cpu().numpy()
        logger.debug(f"    Object encoding shape: {F_Object_T.shape}")

        # Relation branch
        F_Relation_T = relation_model.encode_text(batch["texts"])
        text_enc_relation[index_offset:index_offset + batch_size] = F_Relation_T.cpu().numpy()
        logger.debug(f"    Relation encoding shape: {F_Relation_T.shape}")

        query_cell_ids.extend(batch["cell_ids"])
        index_offset += batch_size

    logger.info("  ✓ 查询编码完成")
    logger.info(f"    - text_enc_global shape: {text_enc_global.shape}")
    logger.info(f"    - text_enc_object shape: {text_enc_object.shape}")
    logger.info(f"    - text_enc_relation shape: {text_enc_relation.shape}")

    # Encode cells
    logger.info("-" * 60)
    logger.info("Phase B: 开始编码场景单元 (Cells)")
    logger.info(f"  - Cells 总数: {n_cells}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - 预期 batch 数量: {(n_cells + args.batch_size - 1) // args.batch_size}")
    logger.info("-" * 60)
    
    logger.info("  [Global branch] 开始编码场景单元...")
    logger.info("  [Object branch] 开始编码场景单元...")
    logger.info("  [Relation branch] 开始编码场景单元...")
    
    index_offset = 0
    for batch_idx, batch in enumerate(tqdm.tqdm(cells_dataloader, desc="  Cell encoding", leave=False)):
        batch_size = len(batch["cell_ids"])
        logger.debug(f"  Batch {batch_idx}: 处理 {batch_size} 个 cells")
        
        # 统计每个 batch 的 objects 数量
        total_objects = sum(len(b) for b in batch["objects"])
        logger.debug(f"    Batch objects 总数: {total_objects}")

        # Global branch
        F_Global = global_model.encode_objects(batch["objects"], batch["object_points"])
        cell_enc_global[index_offset:index_offset + batch_size] = F_Global.cpu().numpy()
        logger.debug(f"    Global cell encoding shape: {F_Global.shape}")

        # Object branch
        F_Object_P, mask_object = object_model.encode_objects(batch["objects"], batch["object_points"])
        cell_enc_object[index_offset:index_offset + batch_size] = F_Object_P.cpu().numpy()
        cell_mask_object[index_offset:index_offset + batch_size] = mask_object.cpu().numpy()
        logger.debug(f"    Object cell encoding shape: {F_Object_P.shape}, mask shape: {mask_object.shape}")

        # Relation branch
        F_Relation_P, mask_relation = relation_model.encode_objects(batch["objects"], batch["object_points"])
        cell_enc_relation[index_offset:index_offset + batch_size] = F_Relation_P.cpu().numpy()
        cell_mask_relation[index_offset:index_offset + batch_size] = mask_relation.cpu().numpy()
        logger.debug(f"    Relation cell encoding shape: {F_Relation_P.shape}, mask shape: {mask_relation.shape}")

        db_cell_ids.extend(batch["cell_ids"])
        index_offset += batch_size

    logger.info("  ✓ Cells 编码完成")
    logger.info(f"    - cell_enc_global shape: {cell_enc_global.shape}")
    logger.info(f"    - cell_enc_object shape: {cell_enc_object.shape}")
    logger.info(f"    - cell_mask_object shape: {cell_mask_object.shape}")
    logger.info(f"    - cell_enc_relation shape: {cell_enc_relation.shape}")
    logger.info(f"    - cell_mask_relation shape: {cell_mask_relation.shape}")
    
    logger.info("-" * 60)
    logger.info(f"编码汇总:")
    logger.info(f"  - 查询数: {n_queries}, Cell数: {n_cells}")
    logger.info(f"  - 查询 cell IDs 数: {len(query_cell_ids)}")
    logger.info(f"  - 数据库 cell IDs 数: {len(db_cell_ids)}")
    logger.info("=" * 80)
    logger.info("[Step 1/4] 编码完成")
    logger.info("=" * 80)

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


@torch.no_grad()
def compute_weighted_scores(encodings, args, logger):
    """
    计算三分支的加权相似度分数

    Returns:
        combined_scores: [N_queries, N_cells] 加权融合后的分数
        scores_dict: 各分支的原始分数字典
    """
    logger.info("=" * 80)
    logger.info("[Step 2/4] 计算三分支相似度分数")
    logger.info("=" * 80)
    
    n_queries = len(encodings['query_cell_ids'])
    n_cells = len(encodings['db_cell_ids'])
    max_top_k = max(args.top_k)
    
    logger.info(f"查询数: {n_queries}, Cells数: {n_cells}")
    logger.info(f"Top-K 设置: {args.top_k}")
    logger.info(f"权重配置: w_global={args.weight_global}, w_object={args.weight_object}, w_relation={args.weight_relation}")

    scores_dict = {}

    # Global scores
    logger.info("-" * 60)
    logger.info("计算 Global 分支分数...")
    scores_global = encodings['text_enc_global'] @ encodings['cell_enc_global'].T
    scores_dict['global'] = scores_global
    logger.info(f"  Global Score Matrix shape: {scores_global.shape}")
    logger.info(f"  Global Score Range: [{scores_global.min():.4f}, {scores_global.max():.4f}]")
    logger.info(f"  Global Score Mean: {scores_global.mean():.4f}, Std: {scores_global.std():.4f}")
    logger.info(f"  Global Score Percentiles: P25={np.percentile(scores_global, 25):.4f}, P50={np.percentile(scores_global, 50):.4f}, P75={np.percentile(scores_global, 75):.4f}")

    # Object scores
    logger.info("-" * 60)
    logger.info("计算 Object 分支分数 (使用注意力机制)...")
    logger.info(f"  Object Score 计算参数: query_batch_size=1000, cell_batch_size=200")
    scores_object = np.zeros((n_queries, n_cells), dtype=np.float32)
    query_batch_size = 1000
    cell_batch_size = 200

    for q_start in range(0, n_queries, query_batch_size):
        q_end = min(q_start + query_batch_size, n_queries)
        text_obj_batch = torch.from_numpy(encodings['text_enc_object'][q_start:q_end]).float()
        logger.debug(f"    处理查询批次: [{q_start}:{q_end}]")

        for c_start in range(0, n_cells, cell_batch_size):
            c_end = min(c_start + cell_batch_size, n_cells)

            cell_obj_batch = torch.from_numpy(encodings['cell_enc_object'][c_start:c_end]).float()
            cell_obj_T = cell_obj_batch.transpose(-1, -2)
            cell_mask_batch = torch.from_numpy(encodings['cell_mask_object'][c_start:c_end]).float()

            text_exp = text_obj_batch.unsqueeze(1)
            cell_exp = cell_obj_T.unsqueeze(0)
            attention = torch.matmul(text_exp, cell_exp)

            mask = cell_mask_batch.unsqueeze(0).unsqueeze(2)
            attention = attention.masked_fill(mask.squeeze(-1) == 0, -100)
            scores_object[q_start:q_end, c_start:c_end] = attention.max(dim=-1)[0].mean(dim=-1).numpy()

    scores_dict['object'] = scores_object
    logger.info(f"  Object Score Matrix shape: {scores_object.shape}")
    logger.info(f"  Object Score Range: [{scores_object.min():.4f}, {scores_object.max():.4f}]")
    logger.info(f"  Object Score Mean: {scores_object.mean():.4f}, Std: {scores_object.std():.4f}")

    # Relation scores
    logger.info("-" * 60)
    logger.info("计算 Relation 分支分数 (使用注意力机制)...")
    scores_relation = np.zeros((n_queries, n_cells), dtype=np.float32)

    for q_start in range(0, n_queries, query_batch_size):
        q_end = min(q_start + query_batch_size, n_queries)
        text_rel_batch = torch.from_numpy(encodings['text_enc_relation'][q_start:q_end]).float()
        logger.debug(f"    处理查询批次: [{q_start}:{q_end}]")

        for c_start in range(0, n_cells, cell_batch_size):
            c_end = min(c_start + cell_batch_size, n_cells)

            cell_rel_batch = torch.from_numpy(encodings['cell_enc_relation'][c_start:c_end]).float()
            cell_rel_T = cell_rel_batch.transpose(-1, -2)
            cell_mask_rel_batch = torch.from_numpy(encodings['cell_mask_relation'][c_start:c_end]).float()

            text_exp = text_rel_batch.unsqueeze(1)
            cell_exp = cell_rel_T.unsqueeze(0)
            attention = torch.matmul(text_exp, cell_exp)

            mask = cell_mask_rel_batch.unsqueeze(0).unsqueeze(2)
            attention = attention.masked_fill(mask.squeeze(-1) == 0, -100)
            scores_relation[q_start:q_end, c_start:c_end] = attention.max(dim=-1)[0].mean(dim=-1).numpy()

    scores_dict['relation'] = scores_relation
    logger.info(f"  Relation Score Matrix shape: {scores_relation.shape}")
    logger.info(f"  Relation Score Range: [{scores_relation.min():.4f}, {scores_relation.max():.4f}]")
    logger.info(f"  Relation Score Mean: {scores_relation.mean():.4f}, Std: {scores_relation.std():.4f}")

    # Weighted combination
    logger.info("-" * 60)
    logger.info("计算加权组合分数...")
    logger.info(f"  [权重组合公式]")
    logger.info(f"  Combined Score = {args.weight_global:.2f} × Global + {args.weight_object:.2f} × Object + {args.weight_relation:.2f} × Relation")
    combined_scores = (
        args.weight_global * scores_global +
        args.weight_object * scores_object +
        args.weight_relation * scores_relation
    )
    logger.info(f"  Combined Score Matrix shape: {combined_scores.shape}")
    logger.info(f"  Combined Score Range: [{combined_scores.min():.4f}, {combined_scores.max():.4f}]")
    logger.info(f"  Combined Score Mean: {combined_scores.mean():.4f}, Std: {combined_scores.std():.4f}")
    
    # 各分支分数的贡献度分析
    logger.info("-" * 60)
    logger.info("各分支分数贡献度分析:")
    logger.info(f"  Global 分数贡献: 范围 [{args.weight_global * scores_global.min():.4f}, {args.weight_global * scores_global.max():.4f}]")
    logger.info(f"  Object 分数贡献: 范围 [{args.weight_object * scores_object.min():.4f}, {args.weight_object * scores_object.max():.4f}]")
    logger.info(f"  Relation 分数贡献: 范围 [{args.weight_relation * scores_relation.min():.4f}, {args.weight_relation * scores_relation.max():.4f}]")
    
    logger.info("=" * 80)
    logger.info("[Step 2/4] 分数计算完成")
    logger.info("=" * 80)

    return combined_scores, scores_dict


def run_coarse_evaluation(combined_scores, encodings, cells_dataset, query_poses_w, args, logger):
    """
    运行 Coarse 阶段的评估，获取 Top-K retrievals

    Returns:
        retrievals: List of retrievals for each query
        retrieval_accuracies: Dict {k: accuracy} - Hit@K 准确率
        retrieval_accuracies_close: Dict {k: accuracy} - Close@K 准确率
    """
    logger.info("=" * 80)
    logger.info("[Step 3/4] Coarse 阶段评估 - 检索 Top-K Cells")
    logger.info("=" * 80)
    
    query_cell_ids = encodings['query_cell_ids']
    db_cell_ids = encodings['db_cell_ids']
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size
    max_top_k = max(args.top_k)
    
    logger.info(f"查询数: {len(query_cell_ids)}")
    logger.info(f"数据库 Cells 数: {len(db_cell_ids)}")
    logger.info(f"Cell size: {cell_size}")
    logger.info(f"Top-K 设置: {args.top_k}")

    retrievals = []
    accuracies = {k: [] for k in args.top_k}
    accuracies_close = {k: [] for k in args.top_k}
    
    # 用于统计的变量
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0

    for query_idx in range(len(query_cell_ids)):
        sorted_indices = np.argsort(-combined_scores[query_idx])
        target_cell_id = query_cell_ids[query_idx]
        retrieved_cell_ids = db_cell_ids[sorted_indices[:max_top_k]]

        retrievals.append(retrieved_cell_ids)
        
        # 记录每个查询的检索结果
        logger.debug(f"Query {query_idx}: target={target_cell_id}, top-1={retrieved_cell_ids[0]}, top-5={retrieved_cell_ids[:5].tolist()}")

        # Calculate Hit@K
        for k in args.top_k:
            top_k_cells = retrieved_cell_ids[:k]
            hit = target_cell_id in top_k_cells
            accuracies[k].append(hit)
            if k == 1 and hit:
                top1_correct += 1
            if k == 5 and hit:
                top5_correct += 1
            if k == 10 and hit:
                top10_correct += 1

        # Calculate Close@K (position-based)
        if query_poses_w is not None:
            target_pose_w = query_poses_w[query_idx]
            retrieved_centers = np.array([
                cells_dict[cid].get_center()[0:2] for cid in retrieved_cell_ids
            ])
            dists = np.linalg.norm(retrieved_centers - target_pose_w, axis=1)

            for k in args.top_k:
                close = np.any(dists[:k] <= cell_size / 2)
                accuracies_close[k].append(close)
            
            logger.debug(f"Query {query_idx}: target_pos={target_pose_w}, distances={dists[:5].tolist()}")

    # Average accuracies - 保持与 pipeline.py 一致的格式
    retrieval_accuracies = {k: np.mean(accuracies[k]) for k in args.top_k}
    retrieval_accuracies_close = {k: np.mean(accuracies_close[k]) for k in args.top_k}
    
    logger.info("-" * 60)
    logger.info("Coarse 阶段检索结果统计:")
    logger.info(f"  Total queries: {len(query_cell_ids)}")
    logger.info(f"  Top-1 命中: {top1_correct} ({100.0 * top1_correct / len(query_cell_ids):.2f}%)")
    logger.info(f"  Top-5 命中: {top5_correct} ({100.0 * top5_correct / len(query_cell_ids):.2f}%)")
    logger.info(f"  Top-10 命中: {top10_correct} ({100.0 * top10_correct / len(query_cell_ids):.2f}%)")
    
    logger.info("-" * 60)
    logger.info("Hit@K 准确率:")
    for k in args.top_k:
        logger.info(f"  Hit@{k}: {retrieval_accuracies[k]:.4f} ({retrieval_accuracies[k] * 100:.2f}%)")
    
    if query_poses_w is not None:
        logger.info("-" * 60)
        logger.info("Close@K 准确率 (位置距离 <= cell_size/2):")
        for k in args.top_k:
            logger.info(f"  Close@{k}: {retrieval_accuracies_close[k]:.4f} ({retrieval_accuracies_close[k] * 100:.2f}%)")
    
    logger.info("=" * 80)
    logger.info("[Step 3/4] Coarse 阶段评估完成")
    logger.info("=" * 80)

    return retrievals, retrieval_accuracies, retrieval_accuracies_close


@torch.no_grad()
def run_fine_evaluation(model_fine, retrievals, all_poses, all_cells, query_subset, args, transform_fine, device, logger):
    """
    运行 Fine 阶段的评估，在 Top-K Cells 中预测精确位置

    Returns:
        accuracies_offset: Dict {k: Dict{thresh: accuracy}} - 各阈值各Top-K的精细定位准确率
    """
    logger.info("=" * 80)
    logger.info("[Step 4/4] Fine 阶段评估 - 精细位置预测")
    logger.info("=" * 80)
    
    model_fine.eval()
    
    dataset_topk = Kitti360TopKDataset(
        all_poses,
        all_cells,
        retrievals,
        transform_fine,
        args,
    )
    
    logger.info(f"Fine 阶段查询数: {len(dataset_topk)}")
    logger.info(f"Top-K 设置: {args.top_k}")
    logger.info(f"阈值设置: {args.threshs}")
    logger.info(f"Device: {device}")

    # Run fine model
    offsets = []
    cell_ids = []
    poses_w = []

    logger.info("-" * 60)
    logger.info("开始 Fine 模型推理...")
    
    pbar = tqdm.tqdm(enumerate(dataset_topk), total=len(dataset_topk), desc="  Fine matching", leave=False)

    for i_sample, sample in pbar:
        texts = sample["texts"]
        output = model_fine(sample["objects"], texts, sample["object_points"])
        offsets.append(output.detach().cpu().numpy())
        cell_ids.append([cell.id for cell in sample["cells"]])
        poses_w.append(sample["poses"][0].pose_w)
        
        # 每100个样本记录一次进度
        if i_sample > 0 and i_sample % 100 == 0:
            logger.debug(f"    已处理 {i_sample}/{len(dataset_topk)} 个样本")

    offsets = np.array(offsets)
    cell_ids = np.array(cell_ids)
    
    logger.info(f"  ✓ Fine 模型推理完成")
    logger.info(f"  Offsets shape: {offsets.shape}")
    logger.info(f"  平均 offset 范数: {np.linalg.norm(offsets, axis=-1).mean():.4f}")

    # Calculate Fine accuracies for all thresholds
    all_cells_dict = {cell.id: cell for cell in all_cells}
    accuracies_offset = {k: {t: [] for t in args.threshs} for k in args.top_k}
    sample_offsets_all = []

    logger.info("-" * 60)
    logger.info("计算精细定位准确率...")
    
    for i_sample in range(len(retrievals)):
        pose = query_subset[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        sample_offsets = offsets[i_sample]
        sample_offsets_all.append(sample_offsets)

        # Get objects and offsets for each top-cell
        pos_in_cells_offsets = []
        for i_cell in range(len(top_cells)):
            cell = deepcopy(top_cells[i_cell])
            while len(cell.objects) < args.pad_size:
                cell.objects.append(Object3d.create_padding())
            cell_offsets = sample_offsets[i_cell]
            pos_in_cells_offsets.append(cell_offsets)
        pos_in_cells_offsets = np.array(pos_in_cells_offsets)

        # Calculate sample accuracies for all thresholds
        accs_offsets = calc_sample_accuracies(
            pose, top_cells, pos_in_cells_offsets, args.top_k, args.threshs
        )

        for k in args.top_k:
            for thresh in args.threshs:
                accuracies_offset[k][thresh].append(accs_offsets[k][thresh])
        
        # 每100个样本记录一次进度
        if i_sample > 0 and i_sample % 100 == 0:
            logger.debug(f"    已计算 {i_sample}/{len(retrievals)} 个样本的准确率")

    # Average accuracies
    for k in args.top_k:
        for thresh in args.threshs:
            accuracies_offset[k][thresh] = np.mean(accuracies_offset[k][thresh])
    
    logger.info("-" * 60)
    logger.info("Fine 阶段精细定位结果:")
    for k in args.top_k:
        logger.info(f"  Top-{k}:")
        for thresh in args.threshs:
            acc = accuracies_offset[k][thresh]
            logger.info(f"    @{thresh}m: {acc:.4f} ({acc * 100:.2f}%)")
    
    logger.info("=" * 80)
    logger.info("[Step 4/4] Fine 阶段评估完成")
    logger.info("=" * 80)

    return accuracies_offset, np.array(sample_offsets_all, dtype=object)


def main():
    import argparse
    parser = create_base_parser()
    add_three_branch_args(parser)
    args = parser.parse_args()
    args = EasyDict(vars(args))

    # 设置日志系统
    logger, log_path = setup_logging(args)
    
    dataset_name = 'TEST' if args.use_test_set else 'VAL'

    logger.info("=" * 80)
    logger.info("三分支 Coarse + Fine 联合评估开始")
    logger.info(f"数据集: {dataset_name}")
    logger.info("=" * 80)
    
    # 记录所有配置参数
    logger.info("=" * 80)
    logger.info("【配置参数】")
    logger.info("=" * 80)
    logger.info(f"Checkpoint 路径:")
    logger.info(f"  - Global:      {args.global_checkpoint}")
    logger.info(f"  - Object:      {args.object_checkpoint}")
    logger.info(f"  - Relation:    {args.relation_checkpoint}")
    logger.info(f"  - Fine:        {args.path_fine}")
    logger.info(f"  ")
    logger.info(f"分支权重:")
    logger.info(f"  - weight_global:    {args.weight_global}")
    logger.info(f"  - weight_object:    {args.weight_object}")
    logger.info(f"  - weight_relation:  {args.weight_relation}")
    logger.info(f"  ")
    logger.info(f"数据路径: {args.base_path}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Thresholds: {args.threshs}")
    logger.info(f"Coarse embed dim: {args.coarse_embed_dim}")
    logger.info(f"PointNet num points: {args.pointnet_numpoints}")
    logger.info(f"Object size: {args.object_size}")
    logger.info(f"Num mentioned: {args.num_mentioned}")
    logger.info(f"使用特征: {args.use_features}")
    logger.info(f"语言模型: {args.hungging_model}")
    logger.info(f"PointNet 路径: {args.pointnet_path}")
    logger.info("=" * 80)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"计算设备: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create transforms
    if args.no_pc_augment:
        transform = T.FixedPoints(args.pointnet_numpoints)
        logger.debug("Coarse transform: FixedPoints (无数据增强)")
    else:
        transform = T.Compose([
            T.FixedPoints(args.pointnet_numpoints),
            T.NormalizeScale()
        ])
        logger.debug("Coarse transform: FixedPoints + NormalizeScale")

    if args.no_pc_augment_fine:
        transform_fine = T.FixedPoints(args.pointnet_numpoints)
        logger.debug("Fine transform: FixedPoints (无数据增强)")
    else:
        transform_fine = T.Compose([
            T.FixedPoints(args.pointnet_numpoints),
            T.NormalizeScale()
        ])
        logger.debug("Fine transform: FixedPoints + NormalizeScale")

    # Create datasets
    logger.info("-" * 60)
    logger.info("加载数据集...")
    
    if args.use_test_set:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_TEST, transform,
            shuffle_hints=False, flip_poses=False,
        )
        logger.info(f"使用 TEST 数据集，场景数: {len(SCENE_NAMES_TEST)}")
    else:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_VAL, transform,
            shuffle_hints=False, flip_poses=False,
        )
        logger.info(f"使用 VAL 数据集，场景数: {len(SCENE_NAMES_VAL)}")

    dataloader_retrieval = DataLoader(
        dataset_retrieval,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    cells_dataset = dataset_retrieval.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    
    logger.info(f"查询 (Query) 数量: {len(dataloader_retrieval.dataset)}")
    logger.info(f"场景单元 (Cell) 数量: {len(cells_dataloader.dataset)}")
    logger.info(f"查询 Poses 数量: {len(dataset_retrieval.all_poses)}")

    # Load three branch models
    logger.info("=" * 80)
    logger.info("[Step 1/4] 加载三分支模型")
    logger.info("=" * 80)
    logger.info(f"加载 Global branch: {osp.basename(args.global_checkpoint)}")
    logger.info(f"加载 Object branch: {osp.basename(args.object_checkpoint)}")
    logger.info(f"加载 Relation branch: {osp.basename(args.relation_checkpoint)}")

    model_global = GlobalBranch(KNOWN_CLASS, COLOR_NAMES_K360, args)
    model_global.load_state_dict(torch.load(args.global_checkpoint, map_location='cpu'), strict=False)
    model_global.to(device).eval()
    logger.info(f"  ✓ Global branch 加载成功, 设备: {next(model_global.parameters()).device}")

    model_object = ObjectBranch(KNOWN_CLASS, COLOR_NAMES_K360, args)
    model_object.load_state_dict(torch.load(args.object_checkpoint, map_location='cpu'), strict=False)
    model_object.to(device).eval()
    logger.info(f"  ✓ Object branch 加载成功, 设备: {next(model_object.parameters()).device}")

    model_relation = RelationBranch(KNOWN_CLASS, COLOR_NAMES_K360, args)
    model_relation.load_state_dict(torch.load(args.relation_checkpoint, map_location='cpu'), strict=False)
    model_relation.to(device).eval()
    logger.info(f"  ✓ Relation branch 加载成功, 设备: {next(model_relation.parameters()).device}")

    logger.info("三分支模型全部加载成功")

    # Load Fine model
    logger.info("=" * 80)
    logger.info("[Step 2/4] 加载 Fine 模型")
    logger.info("=" * 80)
    logger.info(f"加载 Fine 模型: {osp.basename(args.path_fine)}")
    model_fine = CrossMatch(KNOWN_CLASS, COLOR_NAMES_K360, args)
    model_fine_dic = torch.load(args.path_fine, map_location='cpu')
    model_fine.load_state_dict(model_fine_dic, strict=False)
    model_fine.to(device)
    logger.info(f"  ✓ Fine 模型加载成功, 设备: {next(model_fine.parameters()).device}")

    # Restrict to first N queries if requested
    n_vis = min(len(dataset_retrieval.all_poses), int(args.vis_num_queries))
    query_subset = dataset_retrieval.all_poses[:n_vis]
    query_poses_w = np.array([pose.pose_w[0:2] for pose in query_subset])
    logger.debug(f"查询姿态坐标形状: {query_poses_w.shape}")

    dataloader_retrieval = DataLoader(
        torch.utils.data.Subset(dataset_retrieval, range(n_vis)),
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    logger.info(f"仅推理前 {n_vis} 条 query")

    # Encode all branches
    encodings = encode_all_branches(
        model_global, model_object, model_relation,
        dataloader_retrieval, cells_dataloader, args, logger
    )
    encodings['cells_dataset'] = cells_dataset
    encodings['query_poses_w'] = query_poses_w

    # Compute weighted scores
    combined_scores, scores_dict = compute_weighted_scores(encodings, args, logger)

    # Run Coarse evaluation
    retrievals, retrieval_accuracies, retrieval_accuracies_close = run_coarse_evaluation(
        combined_scores, encodings, cells_dataset, query_poses_w, args, logger
    )

    # Run Fine evaluation
    accuracies_offset, sample_offsets = run_fine_evaluation(
        model_fine, retrievals, dataset_retrieval.all_poses[:n_vis], cells_dataset.cells, query_subset, args, transform_fine, device, logger
    )

    # 保存可视化检索结果
    if args.save_retrieval_json:
        from evaluation.visualize_three_branch_retrieval import save_records

        viz_records = build_viz_records(
            query_subset,
            dataset_retrieval.all_cells,
            retrievals,
            combined_scores,
            sample_offsets,
            args,
        )
        save_records(viz_records, args.save_retrieval_json, metadata={
            'base_path': args.base_path,
            'dataset': dataset_name,
            'top_k': args.top_k,
            'threshs': args.threshs,
            'weight_global': args.weight_global,
            'weight_object': args.weight_object,
            'weight_relation': args.weight_relation,
        })
        logger.info(f"可视化结果已保存至: {args.save_retrieval_json}")

    # 打印最终结果
    logger.info("=" * 80)
    logger.info("【最终评估结果汇总】")
    logger.info("=" * 80)
    logger.info("-" * 60)
    logger.info("Coarse 阶段 - 检索准确率 (Hit@K):")
    for k, v in retrieval_accuracies.items():
        logger.info(f"  Hit@{k}: {v:.4f} ({v * 100:.2f}%)")
    
    logger.info("-" * 60)
    logger.info("Coarse 阶段 - 位置准确率 (Close@K):")
    for k, v in retrieval_accuracies_close.items():
        logger.info(f"  Close@{k}: {v:.4f} ({v * 100:.2f}%)")
    
    logger.info("-" * 60)
    logger.info("Fine 阶段 - 精细定位准确率:")
    for k in args.top_k:
        for thresh in args.threshs:
            acc = accuracies_offset[k][thresh]
            logger.info(f"  Top-{k} @{thresh}m: {acc:.4f} ({acc * 100:.2f}%)")
    
    logger.info("=" * 80)
    logger.info("评估完成!")
    logger.info(f"详细日志已保存至: {log_path}")
    logger.info("=" * 80)
    
    # 同时在控制台打印最终结果（与原脚本兼容）
    print()
    print("Retrieval Accs:")
    print(retrieval_accuracies)
    print("Retrieval Accs Close:")
    print(retrieval_accuracies_close)
    print()
    coarse_acc_dict = {k: {args.threshs[0]: v} for k, v in retrieval_accuracies.items()}
    print_accuracies(coarse_acc_dict, "Coarse")
    print()
    print_accuracies(accuracies_offset, "Fine")
    print()
    print(f"详细日志已保存至: {log_path}")


if __name__ == "__main__":
    main()
