"""
三分支权重网格搜索评估脚本

对 Global、Object、Relation 三个分支进行权重网格搜索，找到最优的权重组合。

使用方式:
    python -m evaluation.pipeline_weight_search \
        --global_checkpoint checkpoints/global/global_epoch21_acc0.7841.pth \
        --object_checkpoint checkpoints/object/object_epoch15_acc0.8754.pth \
        --relation_checkpoint checkpoints/relation/relation_epoch14_acc0.8701.pth \
        --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
        --batch_size 64 \
        --coarse_embed_dim 256 \
        --no_pc_augment \
        --fixed_embedding \
        --use_features class color position num \
        --hungging_model t5-large \
        --pointnet_path ./checkpoints/pointnet_acc0.86_lr1_p256.pth \
        --num_mentioned 6 \
        --object_size 28 \
        --inter_module_num_heads 4 \
        --inter_module_num_layers 1 \
        --intra_module_num_heads 4 \
        --intra_module_num_layers 1 \
        --num_of_hidden_layer 3 \
        --alpha 2 \
        --weight_step 0.1 \
        --use_test_set \
        --output_dir ./weight_search_results
"""

import torch
import numpy as np
import os.path as osp
import os
import json
import csv
from datetime import datetime
from easydict import EasyDict
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
import tqdm

from models.branches import GlobalBranch, ObjectBranch, RelationBranch
from evaluation.args import create_base_parser
from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from datapreparation.kitti360pose.utils import SCENE_NAMES_TEST, SCENE_NAMES_VAL, KNOWN_CLASS
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360


def add_weight_search_args(parser):
    """添加权重搜索相关的命令行参数"""
    parser.add_argument('--weight_step', type=float, default=0.1,
                       help='Weight search step size (default: 0.05)')
    parser.add_argument('--weight_range_min', type=float, default=0.0,
                       help='Minimum weight value (default: 0.0)')
    parser.add_argument('--weight_range_max', type=float, default=1.0,
                       help='Maximum weight value (default: 1.0)')
    parser.add_argument('--sort_metric', type=str, default='hit@1',
                       choices=['hit@1', 'hit@3', 'hit@5', 'hit@10', 'close@1', 'close@3', 'close@5', 'close@10'],
                       help='Metric to sort results (default: hit@1)')
    parser.add_argument('--top_n_results', type=int, default=20,
                       help='Number of top results to display (default: 20)')
    parser.add_argument('--output_dir', type=str, default='./weight_search_results',
                       help='Directory to save results (default: ./weight_search_results)')
    parser.add_argument('--global_checkpoint', type=str, required=True,
                       help='Path to Global branch checkpoint')
    parser.add_argument('--object_checkpoint', type=str, required=True,
                       help='Path to Object branch checkpoint')
    parser.add_argument('--relation_checkpoint', type=str, required=True,
                       help='Path to Relation branch checkpoint')


@torch.no_grad()
def encode_all_branches(global_model, object_model, relation_model, dataloader, cells_dataloader, args):
    """使用三个分支分别编码查询和场景"""
    n_queries = len(dataloader.dataset)
    n_cells = len(cells_dataloader.dataset)
    embed_dim = args.coarse_embed_dim
    
    text_enc_global = np.zeros((n_queries, embed_dim), dtype=np.float32)
    cell_enc_global = np.zeros((n_cells, embed_dim), dtype=np.float32)
    text_enc_object = np.zeros((n_queries, args.num_mentioned, embed_dim), dtype=np.float32)
    cell_enc_object = np.zeros((n_cells, args.object_size, embed_dim), dtype=np.float32)
    cell_mask_object = np.zeros((n_cells, args.object_size, 1), dtype=np.float32)
    text_enc_relation = np.zeros((n_queries, args.num_mentioned, embed_dim), dtype=np.float32)
    cell_enc_relation = np.zeros((n_cells, args.object_size, embed_dim), dtype=np.float32)
    cell_mask_relation = np.zeros((n_cells, args.object_size, 1), dtype=np.float32)
    
    query_cell_ids = []
    db_cell_ids = []

    print("\n  [1/2] Encoding queries...")
    index_offset = 0
    for batch in tqdm.tqdm(dataloader, desc="  Query encoding", leave=False):
        batch_size = len(batch["texts"])
        text_enc_global[index_offset:index_offset + batch_size] = global_model.encode_text(batch["texts"]).cpu().numpy()
        text_enc_object[index_offset:index_offset + batch_size] = object_model.encode_text(batch["texts"]).cpu().numpy()
        text_enc_relation[index_offset:index_offset + batch_size] = relation_model.encode_text(batch["texts"]).cpu().numpy()
        query_cell_ids.extend(batch["cell_ids"])
        index_offset += batch_size

    print("\n  [2/2] Encoding cells...")
    index_offset = 0
    for batch in tqdm.tqdm(cells_dataloader, desc="  Cell encoding", leave=False):
        batch_size = len(batch["cell_ids"])
        cell_enc_global[index_offset:index_offset + batch_size] = global_model.encode_objects(batch["objects"], batch["object_points"]).cpu().numpy()
        F_Object_P, mask_object = object_model.encode_objects(batch["objects"], batch["object_points"])
        cell_enc_object[index_offset:index_offset + batch_size] = F_Object_P.cpu().numpy()
        cell_mask_object[index_offset:index_offset + batch_size] = mask_object.cpu().numpy()
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


def compute_branch_scores(encodings, args):
    """计算三个分支的原始相似度分数（分批计算以节省内存）"""
    scores_dict = {}
    n_queries = len(encodings['query_cell_ids'])
    n_cells = len(encodings['db_cell_ids'])
    
    # Global scores: 直接矩阵乘法
    scores_global = encodings['text_enc_global'] @ encodings['cell_enc_global'].T
    scores_dict['global'] = scores_global
    
    # 同时分批处理 queries 和 cells
    query_batch_size = 1000  # 每批处理 1000 个 queries
    cell_batch_size = 200     # 每批处理 200 个 cells
    
    scores_object = np.zeros((n_queries, n_cells), dtype=np.float32)
    scores_relation = np.zeros((n_queries, n_cells), dtype=np.float32)
    
    print(f"    Computing scores in query batches of {query_batch_size}, cell batches of {cell_batch_size}...")
    
    for q_start in range(0, n_queries, query_batch_size):
        q_end = min(q_start + query_batch_size, n_queries)
        print(f"      Processing queries {q_start}-{q_end}/{n_queries}...")
        
        # 获取当前批次的 query 数据
        text_obj_batch = torch.from_numpy(encodings['text_enc_object'][q_start:q_end]).float()  # [q_batch, M, D]
        text_rel_batch = torch.from_numpy(encodings['text_enc_relation'][q_start:q_end]).float()
        
        for c_start in range(0, n_cells, cell_batch_size):
            c_end = min(c_start + cell_batch_size, n_cells)
            
            # 获取当前批次的 cell 数据
            cell_obj_batch = torch.from_numpy(encodings['cell_enc_object'][c_start:c_end]).float()  # [c_batch, O, D]
            cell_mask_batch = torch.from_numpy(encodings['cell_mask_object'][c_start:c_end]).float()  # [c_batch, O, 1]
            cell_rel_batch = torch.from_numpy(encodings['cell_enc_relation'][c_start:c_end]).float()
            cell_mask_rel_batch = torch.from_numpy(encodings['cell_mask_relation'][c_start:c_end]).float()
            
            # 转置 cell 数据: [c_batch, O, D] -> [c_batch, D, O]
            cell_obj_T = cell_obj_batch.transpose(-1, -2)  # [c_batch, D, O]
            cell_rel_T = cell_rel_batch.transpose(-1, -2)
            
            q_batch_size = q_end - q_start
            c_batch_size = c_end - c_start
            
            # Object scores
            text_exp = text_obj_batch.unsqueeze(1)  # [q_batch, 1, M, D]
            cell_exp = cell_obj_T.unsqueeze(0)      # [1, c_batch, D, O]
            attention = torch.matmul(text_exp, cell_exp)  # [q_batch, c_batch, M, O]
            
            # 应用 mask
            mask = cell_mask_batch.unsqueeze(0).unsqueeze(2)  # [1, c_batch, 1, O, 1]
            attention = attention.masked_fill(mask.squeeze(-1) == 0, -100)
            scores_object[q_start:q_end, c_start:c_end] = attention.max(dim=-1)[0].mean(dim=-1).numpy()
            
            # Relation scores
            text_rel_exp = text_rel_batch.unsqueeze(1)
            attention_rel = torch.matmul(text_rel_exp, cell_rel_T.unsqueeze(0))
            mask_rel = cell_mask_rel_batch.unsqueeze(0).unsqueeze(2)
            attention_rel = attention_rel.masked_fill(mask_rel.squeeze(-1) == 0, -100)
            scores_relation[q_start:q_end, c_start:c_end] = attention_rel.max(dim=-1)[0].mean(dim=-1).numpy()
            
            # 释放显存
            del attention, attention_rel, cell_obj_batch, cell_rel_batch
            del cell_obj_T, cell_rel_T, cell_mask_batch, cell_mask_rel_batch
            del text_exp, cell_exp, mask, mask_rel
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    scores_dict['object'] = scores_object
    scores_dict['relation'] = scores_relation
    
    return scores_dict


class FastEvaluator:
    """快速评估器"""
    
    def __init__(self, scores_dict, query_cell_ids, db_cell_ids, cells_dataset, query_poses_w, top_k):
        self.scores_global = scores_dict['global']  # [N_q, N_c]
        self.scores_object = scores_dict['object']
        self.scores_relation = scores_dict['relation']
        
        self.n_queries = len(query_cell_ids)
        self.n_cells = len(db_cell_ids)
        self.query_cell_ids = query_cell_ids
        self.db_cell_ids = db_cell_ids
        self.cells_dict = {cell.id: cell for cell in cells_dataset.cells}
        self.cell_size = cells_dataset.cells[0].cell_size
        self.query_poses_w = query_poses_w
        self.top_k = top_k
        
        # 预计算target位置用于Close@K
        if query_poses_w is not None:
            self.target_poses = query_poses_w[:, :2]  # [N_q, 2]
            # 预计算所有cell的中心位置
            self.cell_centers = np.array([
                self.cells_dict[cid].get_center()[0:2] for cid in db_cell_ids
            ])  # [N_c, 2]
        
        # 预排序 indices 用于加速
        self.sorted_indices_global = np.argsort(-self.scores_global, axis=1)
    
    def evaluate_single(self, w_g, w_o, w_r):
        """评估单个权重组合"""
        # 计算组合分数
        combined_scores = (
            w_g * self.scores_global + 
            w_o * self.scores_object + 
            w_r * self.scores_relation
        )
        
        # 排序获取 top cells
        sorted_indices = np.argsort(-combined_scores, axis=1)
        retrieved_cell_ids = self.db_cell_ids[sorted_indices]
        
        target_cell_ids = self.query_cell_ids
        
        result = {
            'w_global': float(w_g),
            'w_object': float(w_o),
            'w_relation': float(w_r),
        }
        
        # 计算 Hit@K 和 Close@K
        for k in self.top_k:
            top_k_cells = retrieved_cell_ids[:, :k]
            hits = np.any(top_k_cells == target_cell_ids[:, np.newaxis], axis=1)
            result[f'hit@{k}'] = float(np.mean(hits))
            
            if self.query_poses_w is not None:
                top_k_indices = sorted_indices[:, :k]
                retrieved_centers = self.cell_centers[top_k_indices]
                dists = np.linalg.norm(retrieved_centers - self.target_poses[:, np.newaxis, :], axis=2)
                close = np.any(dists <= self.cell_size / 2, axis=1)
                result[f'close@{k}'] = float(np.mean(close))
        
        return result


def grid_search_weights(scores_dict, encodings, cells_dataset, query_poses_w, args):
    """网格搜索最优权重组合"""
    weights = np.arange(args.weight_range_min, args.weight_range_max + 1e-8, args.weight_step)
    weights = np.round(weights, 2)
    
    total_combinations = sum(1 for w_g in weights for w_o in weights for w_r in weights 
                              if not (w_g == 0 and w_o == 0 and w_r == 0))
    
    print(f"\n  Weight range: [{args.weight_range_min:.2f}, {args.weight_range_max:.2f}], step: {args.weight_step}")
    print(f"  Valid combinations: {total_combinations}")
    print(f"  Sort metric: {args.sort_metric}")
    
    # 创建评估器
    evaluator = FastEvaluator(
        scores_dict, 
        encodings['query_cell_ids'], 
        encodings['db_cell_ids'],
        cells_dataset,
        query_poses_w,
        args.top_k
    )
    
    # 收集所有有效组合
    all_weights = []
    for w_g in weights:
        for w_o in weights:
            for w_r in weights:
                if w_g == 0 and w_o == 0 and w_r == 0:
                    continue
                all_weights.append((w_g, w_o, w_r))
    
    # 逐个评估（避免内存溢出）
    all_results = []
    pbar = tqdm.tqdm(total=total_combinations, desc="  Weight search", unit="combo")
    
    for w_g, w_o, w_r in all_weights:
        result = evaluator.evaluate_single(w_g, w_o, w_r)
        all_results.append(result)
        pbar.update(1)
    
    pbar.close()
    
    # 排序
    all_results.sort(key=lambda x: x[args.sort_metric], reverse=True)
    
    return all_results


def print_results_table(results, args):
    """打印结果表格"""
    top_n = min(args.top_n_results, len(results))
    
    # 表头
    header1 = f"{'Rank':^5}│{'Global':^8}│{'Object':^8}│{'Relation':^8}"
    header2 = ""
    for k in args.top_k:
        header1 += f"│{f'Hit@{k}':^9}"
    for k in args.top_k:
        header1 += f"│{f'Close@{k}':^9}"
    
    sep1 = "─" * 5 + "┼" + "─" * 8 + "┼" + "─" * 8 + "┼" + "─" * 8
    sep2 = ""
    for k in args.top_k:
        sep1 += "┼" + "─" * 9
        sep2 += "┼" + "─" * 9
    sep1 += "┼" + sep2
    
    print("╔" + sep1.replace("┼", "╤").replace("─", "═") + "╗")
    print("║" + header1 + "║")
    print("╠" + sep1.replace("┼", "╪") + "╣")
    
    for i, result in enumerate(results[:top_n]):
        row = f"{i+1:^5}│{result['w_global']:^8.2f}│{result['w_object']:^8.2f}│{result['w_relation']:^8.2f}"
        for k in args.top_k:
            row += f"│{result[f'hit@{k}']:^9.4f}"
        for k in args.top_k:
            row += f"│{result[f'close@{k}']:^9.4f}"
        print("║" + row + "║")
    
    print("╚" + sep1.replace("┼", "╧").replace("─", "═") + "╝")


def save_results(results, args, output_dir):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_name = 'test' if args.use_test_set else 'val'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # CSV
    csv_path = os.path.join(output_dir, f'weight_search_{dataset_name}_step{args.weight_step}_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\n  ✓ CSV saved: {csv_path}")
    
    # JSON
    json_path = os.path.join(output_dir, f'weight_search_{dataset_name}_step{args.weight_step}_{timestamp}.json')
    json_data = {
        'search_config': {
            'weight_step': args.weight_step,
            'weight_range_min': args.weight_range_min,
            'weight_range_max': args.weight_range_max,
            'sort_metric': args.sort_metric,
            'dataset': dataset_name,
            'total_combinations': len(results),
        },
        'best_weights': {
            'global': results[0]['w_global'],
            'object': results[0]['w_object'],
            'relation': results[0]['w_relation'],
        },
        'best_hit_accuracy': {f'hit@{k}': results[0][f'hit@{k}'] for k in args.top_k},
        'best_close_accuracy': {f'close@{k}': results[0][f'close@{k}'] for k in args.top_k},
        'top_combinations': results[:args.top_n_results],
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  ✓ JSON saved: {json_path}")
    
    return csv_path, json_path


def main():
    import argparse
    
    parser = create_base_parser()
    add_weight_search_args(parser)
    args = parser.parse_args()
    args = EasyDict(vars(args))
    
    dataset_name = 'TEST' if args.use_test_set else 'VAL'
    
    # 计算总组合数
    num_weights = int((args.weight_range_max - args.weight_range_min) / args.weight_step + 1)
    est_combinations = num_weights ** 3 - 1  # 减去全零组合
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              Branch Weight Grid Search - {dataset_name} Dataset                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Search Configuration:                                                        ║
║    Weight Step:          {args.weight_step:.2f}                                               ║
║    Weight Range:         [{args.weight_range_min:.2f}, {args.weight_range_max:.2f}]                                          ║
║    Weight Points:        {num_weights} per branch                                        ║
║    Total Combinations:   {est_combinations} (estimated, excl. all-zero)               ║
║    Sort Metric:          {args.sort_metric}                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Checkpoints:                                                                   ║
║    Global:    {osp.basename(args.global_checkpoint):>50s}        ║
║    Object:    {osp.basename(args.object_checkpoint):>50s}        ║
║    Relation:  {osp.basename(args.relation_checkpoint):>50s}        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}\n")

    # 变换
    if args.no_pc_augment:
        transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform = T.Compose([
            T.FixedPoints(args.pointnet_numpoints),
            T.NormalizeScale()
        ])

    # 数据集
    scene_names = SCENE_NAMES_TEST if args.use_test_set else SCENE_NAMES_VAL
    dataset = Kitti360CoarseDatasetMulti(
        args.base_path, scene_names, transform,
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

    # 加载模型
    print("[1/4] Loading models...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model_global = GlobalBranch(KNOWN_CLASS, COLOR_NAMES_K360, args)
    model_global.load_state_dict(torch.load(args.global_checkpoint, map_location='cpu'), strict=False)
    model_global.to(device).eval()
    
    model_object = ObjectBranch(KNOWN_CLASS, COLOR_NAMES_K360, args)
    model_object.load_state_dict(torch.load(args.object_checkpoint, map_location='cpu'), strict=False)
    model_object.to(device).eval()
    
    model_relation = RelationBranch(KNOWN_CLASS, COLOR_NAMES_K360, args)
    model_relation.load_state_dict(torch.load(args.relation_checkpoint, map_location='cpu'), strict=False)
    model_relation.to(device).eval()
    
    print("  ✓ All models loaded successfully\n")

    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataset.all_poses])

    # 编码
    print("[2/4] Encoding queries and cells...")
    
    # 检查是否有缓存的编码结果
    cache_path = osp.join(args.output_dir, 'encodings_cache.npz')
    if osp.exists(cache_path):
        print(f"  Loading cached encodings from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        encodings = {
            'text_enc_global': cache['text_enc_global'],
            'cell_enc_global': cache['cell_enc_global'],
            'text_enc_object': cache['text_enc_object'],
            'cell_enc_object': cache['cell_enc_object'],
            'cell_mask_object': cache['cell_mask_object'],
            'text_enc_relation': cache['text_enc_relation'],
            'cell_enc_relation': cache['cell_enc_relation'],
            'cell_mask_relation': cache['cell_mask_relation'],
            'query_cell_ids': cache['query_cell_ids'],
            'db_cell_ids': cache['db_cell_ids'],
        }
        encodings['cells_dataset'] = cells_dataset
        print(f"  ✓ Loaded {len(encodings['query_cell_ids'])} queries, {len(encodings['db_cell_ids'])} cells")
    else:
        encodings = encode_all_branches(
            model_global, model_object, model_relation,
            dataloader, cells_dataloader, args
        )
        encodings['cells_dataset'] = cells_dataset
        
        # 保存编码结果到缓存
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"  Saving encodings cache to {cache_path}")
        np.savez_compressed(
            cache_path,
            text_enc_global=encodings['text_enc_global'],
            cell_enc_global=encodings['cell_enc_global'],
            text_enc_object=encodings['text_enc_object'],
            cell_enc_object=encodings['cell_enc_object'],
            cell_mask_object=encodings['cell_mask_object'],
            text_enc_relation=encodings['text_enc_relation'],
            cell_enc_relation=encodings['cell_enc_relation'],
            cell_mask_relation=encodings['cell_mask_relation'],
            query_cell_ids=encodings['query_cell_ids'],
            db_cell_ids=encodings['db_cell_ids'],
        )
        print(f"  ✓ Encodings cache saved")
    print()

    # 计算分支分数
    print("[3/4] Computing branch similarities...")
    
    # 检查是否有缓存的分数结果
    scores_cache_path = osp.join(args.output_dir, 'scores_cache.npz')
    if osp.exists(scores_cache_path):
        print(f"  Loading cached scores from {scores_cache_path}")
        scores_cache = np.load(scores_cache_path)
        scores_dict = {
            'global': scores_cache['scores_global'],
            'object': scores_cache['scores_object'],
            'relation': scores_cache['scores_relation'],
        }
        print(f"  ✓ Loaded cached scores")
    else:
        scores_dict = compute_branch_scores(encodings, args)
        print(f"  ✓ Global score range:   [{scores_dict['global'].min():.3f}, {scores_dict['global'].max():.3f}]")
        print(f"  ✓ Object score range:    [{scores_dict['object'].min():.3f}, {scores_dict['object'].max():.3f}]")
        print(f"  ✓ Relation score range:  [{scores_dict['relation'].min():.3f}, {scores_dict['relation'].max():.3f}]")
        
        # 保存分数到缓存
        print(f"  Saving scores cache to {scores_cache_path}")
        np.savez_compressed(
            scores_cache_path,
            scores_global=scores_dict['global'],
            scores_object=scores_dict['object'],
            scores_relation=scores_dict['relation'],
        )
        print(f"  ✓ Scores cache saved")
    print()

    # 网格搜索
    print("[4/4] Grid search for optimal weights...")
    start_time = datetime.now()
    results = grid_search_weights(
        scores_dict, encodings, cells_dataset, query_poses_w, args
    )
    search_time = (datetime.now() - start_time).total_seconds()
    print(f"\n  Search completed in {search_time:.1f} seconds\n")

    # 打印结果
    print(f"╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"║                           Top {args.top_n_results} Results                                   ║")
    print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
    print_results_table(results, args)
    print(f"╚══════════════════════════════════════════════════════════════════════════════╝")

    # 保存结果
    print("\n[5/5] Saving results...")
    csv_path, json_path = save_results(results, args, args.output_dir)

    # 打印最佳组合
    best = results[0]
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         Best Weight Combination                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║   Optimal weights:                                                            ║
║     - Global:   {best['w_global']:.2f}                                                           ║
║     - Object:   {best['w_object']:.2f}                                                           ║
║     - Relation: {best['w_relation']:.2f}                                                           ║
║                                                                                ║
║   Accuracy on {dataset_name} set:                                               ║""")
    for k in args.top_k:
        print(f"║     Hit@{k:2d}:  {best[f'hit@{k}']:.4f}                                                           ║")
    for k in args.top_k:
        print(f"║     Close@{k:2d}: {best[f'close@{k}']:.4f}                                                           ║")
    print(f"""║                                                                                ║
║   Results saved to:                                                           ║
║     - {osp.basename(csv_path)}                                                        ║
║     - {osp.basename(json_path)}                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
