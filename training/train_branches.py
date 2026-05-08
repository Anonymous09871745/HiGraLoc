"""
三分支独立训练脚本

支持三种训练模式:
- --branch all: 顺序训练 Global → Object → Relation
- --branch global/object/relation: 单独训练指定分支

每个分支独立保存权重到 checkpoints/{branch}/ 目录。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
import collections
import time
import numpy as np
import os
import os.path as osp
import tqdm
from easydict import EasyDict

from models.branches import GlobalBranch, ObjectBranch, RelationBranch
from datapreparation.kitti360pose.utils import SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti, Kitti360CoarseDataset
from training.args import parse_arguments
from training.losses import CCL, CCL_input_score


def z_score(tensor):
    mean = np.mean(tensor)
    std = np.std(tensor)
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor


def create_save_dir(branch_name):
    """创建分支保存目录"""
    save_dir = f"./checkpoints/{branch_name}"
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def save_training_log(branch_name, epoch, train_loss, val_acc, args, save_dir, is_best=False):
    """保存训练日志到文件"""
    log_file = osp.join(save_dir, f"{branch_name}_training_log.txt")
    
    with open(log_file, 'a') as f:
        if epoch == 1:
            f.write(f"{'='*80}\n")
            f.write(f"Branch: {branch_name.upper()}\n")
            f.write(f"Configuration:\n")
            f.write(f"  - epochs_per_branch: {args.epochs_per_branch}\n")
            f.write(f"  - learning_rate: {args.lr}\n")
            f.write(f"  - batch_size: {args.batch_size}\n")
            f.write(f"  - coarse_embed_dim: {args.coarse_embed_dim}\n")
            f.write(f"  - ranking_loss: {args.ranking_loss}\n")
            f.write(f"  - temperature: {args.temperature}\n")
            f.write(f"  - alpha: {args.alpha}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"{'Epoch':<8} {'TrainLoss':<12} " + 
                    "".join([f"{'Acc@'+str(k):<15}" for k in args.top_k]) +
                    f"{'Best@'+str(max(args.top_k)):<15} {'Status':<10}\n")
            f.write(f"{'-'*80}\n")
        
        best_marker = "★ BEST" if is_best else ""
        f.write(f"{epoch:<8} {train_loss:<12.6f} " +
                "".join([f"{val_acc[k]:<15.4f}" for k in args.top_k]) +
                f"{val_acc[max(args.top_k)]:<15.4f} {best_marker:<10}\n")


def log_header(branch_name, epoch, epochs):
    """打印训练头部日志"""
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  [BRANCH: {branch_name.upper()}] Epoch {epoch}/{epochs}                                  ║
╠══════════════════════════════════════════════════════════════════════╣""")


def log_metrics(branch_name, train_loss, acc_metrics, best_acc, is_best, metrics_type=""):
    """打印训练指标日志"""
    acc_str = "  │  ".join([f"Val Acc@{k}: {v:.4f}" for k, v in acc_metrics.items()])
    best_str = f"Best: {best_acc:.4f} {'★' if is_best else ' '}"
    
    print(f"""║  Train Loss: {train_loss:.4f}                                               ║
║  {acc_str}   ║
║  {best_str}                                                        ║
╚══════════════════════════════════════════════════════════════════════╝""")


# ==================== Global Branch Training ====================

def train_global_epoch(model, dataloader, criterion, optimizer, args, thresh=0.8):
    """训练 Global 分支一个 epoch"""
    model.train()
    epoch_losses = []

    for i_batch, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        optimizer.zero_grad()

        anchor = model.encode_text(batch["texts"])
        F_Global = model.encode_objects(batch["objects"], batch["object_points"])
        
        loss = criterion(anchor, F_Global)

        if torch.isnan(loss).any():
            import ipdb
            ipdb.set_trace()

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        torch.cuda.empty_cache()

    return np.mean(epoch_losses)


@torch.no_grad()
def eval_global_branch(model, dataloader, args):
    """评估 Global 分支，返回准确率"""
    model.eval()
    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    text_encodings = []
    cell_encodings = []
    query_cell_ids = []
    db_cell_ids = []

    # Encode queries
    for batch in dataloader:
        anchor = model.encode_text(batch["texts"])
        text_encodings.append(anchor.cpu().numpy())
        query_cell_ids.extend(batch["cell_ids"])

    # Encode cells
    for batch in cells_dataloader:
        F_Global = model.encode_objects(batch["objects"], batch["object_points"])
        cell_encodings.append(F_Global.cpu().numpy())
        db_cell_ids.extend(batch["cell_ids"])

    text_encodings = np.vstack(text_encodings)
    cell_encodings = np.vstack(cell_encodings)
    query_cell_ids = np.array(query_cell_ids)
    db_cell_ids = np.array(db_cell_ids)

    # Calculate similarities
    scores = text_encodings @ cell_encodings.T
    
    # Calculate accuracy
    accuracies = {k: [] for k in args.top_k}
    for query_idx in range(len(query_cell_ids)):
        sorted_indices = np.argsort(-scores[query_idx])
        target_cell_id = query_cell_ids[query_idx]
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        
        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[:k])

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])

    return accuracies


def train_global_branch(model, train_loader, val_loader, optimizer, epochs, args, save_dir):
    """训练 Global 分支"""
    print(f"\n{'='*70}")
    print(f"[1/3] Training Global Branch")
    print(f"{'='*70}")

    criterion = CCL(temperature=args.temperature, alpha=args.alpha)
    
    best_val_acc = -1
    best_model_path = None
    last_model_path = None

    for epoch in range(1, epochs + 1):
        print(f"\n{'─'*70}")
        log_header("global", epoch, epochs)
        
        train_loss = train_global_epoch(model, train_loader, criterion, optimizer, args)
        val_acc = eval_global_branch(model, val_loader, args)
        
        is_best = val_acc[max(args.top_k)] > best_val_acc
        if is_best:
            best_val_acc = val_acc[max(args.top_k)]
        
        log_metrics("global", train_loss, val_acc, best_val_acc, is_best)

        # Save training log
        save_training_log("global", epoch, train_loss, val_acc, args, save_dir, is_best)

        # Save checkpoint
        model_path = osp.join(save_dir, f"global_epoch{epoch}_acc{val_acc[max(args.top_k)]:.4f}.pth")
        
        if is_best:
            model_dic = model.state_dict()
            out = collections.OrderedDict()
            for item in model_dic:
                if "llm_model" not in item:
                    out[item] = model_dic[item]
            torch.save(out, model_path)
            best_model_path = model_path
            
            if last_model_path and osp.exists(last_model_path) and last_model_path != model_path:
                os.remove(last_model_path)
            last_model_path = model_path
            print(f"  ★ Saved: {model_path}")

    # Save best model
    best_path = osp.join(save_dir, "global_best.pth")
    if best_model_path and best_model_path != best_path:
        torch.save(torch.load(best_model_path), best_path)
        print(f"\n  ★ Best model saved: {best_path}")

    return best_val_acc


# ==================== Object Branch Training ====================

def train_object_epoch(model, dataloader, criterion, optimizer, args, thresh=0.8):
    """训练 Object 分支一个 epoch"""
    model.train()
    epoch_losses = []

    for i_batch, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        optimizer.zero_grad()

        F_Object_T = model.encode_text(batch["texts"])
        F_Object_P, object_level_masks = model.encode_objects(batch["objects"], batch["object_points"])
        batch_size = len(F_Object_T)

        # Calculate similarity score of Object-level
        aa_expanded = F_Object_T.unsqueeze(1).expand(batch_size, batch_size, -1, -1)
        bb_transposed = F_Object_P.transpose(1, 2)
        cc = torch.matmul(aa_expanded, bb_transposed.unsqueeze(0)).squeeze(0)
        aa_mask = torch.ones((batch_size, batch_size, F_Object_T.size(1), 1), device=model.device)
        bb_mask = object_level_masks.transpose(1, 2)
        score_mask = torch.matmul(aa_mask, bb_mask)
        cc[score_mask == 0] = -100
        scores_object = cc.max(dim=-1)[0].mean(dim=-1)

        # Calculate distance
        mean_F_Object_P = F_Object_P.mean(dim=1)
        mean_F_Object_P = mean_F_Object_P / torch.norm(mean_F_Object_P, p=2, dim=1, keepdim=True)
        mean_F_Object_T = F_Object_T.mean(dim=1)
        mean_F_Object_T = mean_F_Object_T / torch.norm(mean_F_Object_T, p=2, dim=1, keepdim=True)
        dist_Object = abs(torch.norm(mean_F_Object_P.unsqueeze(1) - mean_F_Object_T.unsqueeze(0), dim=2))
        dist_Object = dist_Object / dist_Object.max()
        dist_Object = pow((1 - dist_Object + 1e-6), 1 / args.alpha)
        dist_Object[dist_Object > 1] = 1
        dist_Object[dist_Object < thresh] = thresh

        loss = criterion(scores_object, dist_Object)

        if torch.isnan(loss).any():
            import ipdb
            ipdb.set_trace()

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        torch.cuda.empty_cache()

    return np.mean(epoch_losses)


@torch.no_grad()
def eval_object_branch(model, dataloader, args):
    """评估 Object 分支，返回准确率"""
    model.eval()
    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    text_encodings = []
    cell_encodings = []
    cell_masks = []
    query_cell_ids = []
    db_cell_ids = []

    # Encode queries
    for batch in dataloader:
        F_Object_T = model.encode_text(batch["texts"])
        text_encodings.append(F_Object_T.cpu().numpy())
        query_cell_ids.extend(batch["cell_ids"])

    # Encode cells
    for batch in cells_dataloader:
        F_Object_P, object_masks = model.encode_objects(batch["objects"], batch["object_points"])
        cell_encodings.append(F_Object_P.cpu().numpy())
        cell_masks.append(object_masks.cpu().numpy())
        db_cell_ids.extend(batch["cell_ids"])

    text_encodings = np.vstack(text_encodings)
    cell_encodings = np.vstack(cell_encodings)
    cell_masks = np.vstack(cell_masks)
    query_cell_ids = np.array(query_cell_ids)
    db_cell_ids = np.array(db_cell_ids)

    # Calculate similarities
    scores = np.zeros((len(query_cell_ids), len(db_cell_ids)))
    for query_idx in range(len(query_cell_ids)):
        aa = torch.from_numpy(text_encodings[query_idx])
        bb = torch.from_numpy(cell_encodings)
        aa = aa.unsqueeze(0).expand(len(cell_encodings), args.num_mentioned, args.coarse_embed_dim)
        bb = bb.transpose(1, 2)
        cc = torch.matmul(aa, bb)
        aa_mask = torch.ones((args.num_mentioned, 1))
        bb_mask = torch.from_numpy(cell_masks).transpose(1, 2)
        score_mask = torch.matmul(aa_mask, bb_mask)
        cc[score_mask == 0] = -100
        scores[query_idx] = cc.max(dim=-1)[0].mean(dim=-1).numpy()

    # Calculate accuracy
    accuracies = {k: [] for k in args.top_k}
    for query_idx in range(len(query_cell_ids)):
        sorted_indices = np.argsort(-scores[query_idx])
        target_cell_id = query_cell_ids[query_idx]
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        
        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[:k])

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])

    return accuracies


def train_object_branch(model, train_loader, val_loader, optimizer, epochs, args, save_dir):
    """训练 Object 分支"""
    print(f"\n{'='*70}")
    print(f"[2/3] Training Object Branch")
    print(f"{'='*70}")

    criterion = CCL_input_score(temperature=args.temperature, alpha=args.alpha)
    
    best_val_acc = -1
    best_model_path = None
    last_model_path = None

    for epoch in range(1, epochs + 1):
        print(f"\n{'─'*70}")
        log_header("object", epoch, epochs)
        
        train_loss = train_object_epoch(model, train_loader, criterion, optimizer, args)
        val_acc = eval_object_branch(model, val_loader, args)
        
        is_best = val_acc[max(args.top_k)] > best_val_acc
        if is_best:
            best_val_acc = val_acc[max(args.top_k)]
        
        log_metrics("object", train_loss, val_acc, best_val_acc, is_best)

        # Save training log
        save_training_log("object", epoch, train_loss, val_acc, args, save_dir, is_best)

        # Save checkpoint
        model_path = osp.join(save_dir, f"object_epoch{epoch}_acc{val_acc[max(args.top_k)]:.4f}.pth")
        
        if is_best:
            model_dic = model.state_dict()
            out = collections.OrderedDict()
            for item in model_dic:
                if "llm_model" not in item:
                    out[item] = model_dic[item]
            torch.save(out, model_path)
            best_model_path = model_path
            
            if last_model_path and osp.exists(last_model_path) and last_model_path != model_path:
                os.remove(last_model_path)
            last_model_path = model_path
            print(f"  ★ Saved: {model_path}")

    # Save best model
    best_path = osp.join(save_dir, "object_best.pth")
    if best_model_path and best_model_path != best_path:
        torch.save(torch.load(best_model_path), best_path)
        print(f"\n  ★ Best model saved: {best_path}")

    return best_val_acc


# ==================== Relation Branch Training ====================

def train_relation_epoch(model, dataloader, criterion, optimizer, args, thresh=0.8):
    """训练 Relation 分支一个 epoch"""
    model.train()
    epoch_losses = []

    for i_batch, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        optimizer.zero_grad()

        F_Relation_T = model.encode_text(batch["texts"])
        F_Relation_P, relation_level_masks = model.encode_objects(batch["objects"], batch["object_points"])
        batch_size = len(F_Relation_T)

        # Calculate similarity score of Relation-level
        aa_expanded = F_Relation_T.unsqueeze(1).expand(batch_size, batch_size, -1, -1)
        bb_transposed = F_Relation_P.transpose(1, 2)
        cc = torch.matmul(aa_expanded, bb_transposed.unsqueeze(0)).squeeze(0)
        aa_mask = torch.ones((batch_size, batch_size, F_Relation_T.size(1), 1), device=model.device)
        bb_mask = relation_level_masks.transpose(1, 2)
        score_mask = torch.matmul(aa_mask, bb_mask)
        cc[score_mask == 0] = -100
        scores_relation = cc.max(dim=-1)[0].mean(dim=-1)

        # Calculate distance
        mean_F_Relation_P = F_Relation_P.mean(dim=1)
        mean_F_Relation_P = mean_F_Relation_P / torch.norm(mean_F_Relation_P, p=2, dim=1, keepdim=True)
        mean_F_Relation_T = F_Relation_T.mean(dim=1)
        mean_F_Relation_T = mean_F_Relation_T / torch.norm(mean_F_Relation_T, p=2, dim=1, keepdim=True)
        dist_Relation = abs(torch.norm(mean_F_Relation_P.unsqueeze(1) - mean_F_Relation_T.unsqueeze(0), dim=2))
        dist_Relation = dist_Relation / dist_Relation.max()
        dist_Relation = pow((1 - dist_Relation + 1e-6), 1 / args.alpha)
        dist_Relation[dist_Relation > 1] = 1
        dist_Relation[dist_Relation < thresh] = thresh

        loss = criterion(scores_relation, dist_Relation)

        if torch.isnan(loss).any():
            import ipdb
            ipdb.set_trace()

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        torch.cuda.empty_cache()

    return np.mean(epoch_losses)


@torch.no_grad()
def eval_relation_branch(model, dataloader, args):
    """评估 Relation 分支，返回准确率"""
    model.eval()
    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    text_encodings = []
    cell_encodings = []
    cell_masks = []
    query_cell_ids = []
    db_cell_ids = []

    # Encode queries
    for batch in dataloader:
        F_Relation_T = model.encode_text(batch["texts"])
        text_encodings.append(F_Relation_T.cpu().numpy())
        query_cell_ids.extend(batch["cell_ids"])

    # Encode cells
    for batch in cells_dataloader:
        F_Relation_P, relation_masks = model.encode_objects(batch["objects"], batch["object_points"])
        cell_encodings.append(F_Relation_P.cpu().numpy())
        cell_masks.append(relation_masks.cpu().numpy())
        db_cell_ids.extend(batch["cell_ids"])

    text_encodings = np.vstack(text_encodings)
    cell_encodings = np.vstack(cell_encodings)
    cell_masks = np.vstack(cell_masks)
    query_cell_ids = np.array(query_cell_ids)
    db_cell_ids = np.array(db_cell_ids)

    # Calculate similarities
    scores = np.zeros((len(query_cell_ids), len(db_cell_ids)))
    for query_idx in range(len(query_cell_ids)):
        aa = torch.from_numpy(text_encodings[query_idx])
        bb = torch.from_numpy(cell_encodings)
        aa = aa.unsqueeze(0).expand(len(cell_encodings), args.num_mentioned, args.coarse_embed_dim)
        bb = bb.transpose(1, 2)
        cc = torch.matmul(aa, bb)
        aa_mask = torch.ones((args.num_mentioned, 1))
        bb_mask = torch.from_numpy(cell_masks).transpose(1, 2)
        score_mask = torch.matmul(aa_mask, bb_mask)
        cc[score_mask == 0] = -100
        scores[query_idx] = cc.max(dim=-1)[0].mean(dim=-1).numpy()

    # Calculate accuracy
    accuracies = {k: [] for k in args.top_k}
    for query_idx in range(len(query_cell_ids)):
        sorted_indices = np.argsort(-scores[query_idx])
        target_cell_id = query_cell_ids[query_idx]
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        
        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[:k])

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])

    return accuracies


def train_relation_branch(model, train_loader, val_loader, optimizer, epochs, args, save_dir):
    """训练 Relation 分支"""
    print(f"\n{'='*70}")
    print(f"[3/3] Training Relation Branch")
    print(f"{'='*70}")

    criterion = CCL_input_score(temperature=args.temperature, alpha=args.alpha)
    
    best_val_acc = -1
    best_model_path = None
    last_model_path = None

    for epoch in range(1, epochs + 1):
        print(f"\n{'─'*70}")
        log_header("relation", epoch, epochs)
        
        train_loss = train_relation_epoch(model, train_loader, criterion, optimizer, args)
        val_acc = eval_relation_branch(model, val_loader, args)
        
        is_best = val_acc[max(args.top_k)] > best_val_acc
        if is_best:
            best_val_acc = val_acc[max(args.top_k)]
        
        log_metrics("relation", train_loss, val_acc, best_val_acc, is_best)

        # Save training log
        save_training_log("relation", epoch, train_loss, val_acc, args, save_dir, is_best)

        # Save checkpoint
        model_path = osp.join(save_dir, f"relation_epoch{epoch}_acc{val_acc[max(args.top_k)]:.4f}.pth")
        
        if is_best:
            model_dic = model.state_dict()
            out = collections.OrderedDict()
            for item in model_dic:
                if "llm_model" not in item:
                    out[item] = model_dic[item]
            torch.save(out, model_path)
            best_model_path = model_path
            
            if last_model_path and osp.exists(last_model_path) and last_model_path != model_path:
                os.remove(last_model_path)
            last_model_path = model_path
            print(f"  ★ Saved: {model_path}")

    # Save best model
    best_path = osp.join(save_dir, "relation_best.pth")
    if best_model_path and best_model_path != best_path:
        torch.save(torch.load(best_model_path), best_path)
        print(f"\n  ★ Best model saved: {best_path}")

    return best_val_acc


# ==================== Main Training ====================

def add_branch_arguments(parser):
    """添加分支训练相关的命令行参数"""
    parser.add_argument('--branch', type=str, default='all',
                        choices=['global', 'object', 'relation', 'all'],
                        help='Which branch to train (default: all)')
    parser.add_argument('--epochs_per_branch', type=int, default=None,
                        help='Epochs for each branch (default: same as --epochs)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: same as --learning_rate)')


def build_parser():
    """构建完整的参数解析器"""
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Train Branch Models")
    
    # General
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="K360", help="Currently only K360")
    parser.add_argument("--base_path", type=str, help="Root path of Kitti360Pose")
    
    # Model
    parser.add_argument("--use_features", nargs="+", default=["class", "color", "position", "num"])
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--continue_path", type=str, help="Set to continue from a previous checkpoint")
    parser.add_argument("--no_pc_augment", action="store_true")
    
    # Fine
    parser.add_argument("--fine_embed_dim", type=int, default=128)
    parser.add_argument("--offset_lambda", type=float, default=5)
    parser.add_argument("--pmc_prob", type=float, default=0.0)
    parser.add_argument("--pmc_threshold", type=float, default=0.4)
    parser.add_argument("--fine_num_decoder_heads", type=int, default=4)
    parser.add_argument("--fine_num_decoder_layers", type=int, default=2)
    parser.add_argument("--pad_size", type=int, default=16)
    parser.add_argument("--num_mentioned", type=int, default=6)
    parser.add_argument("--describe_by", type=str, default="all")
    
    # Loss
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--a", type=float, default=10)
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--ranking_loss", type=str, default="pairwise")
    
    # Object-encoder / PointNet
    parser.add_argument("--coarse_embed_dim", type=int, default=256)
    parser.add_argument("--pointnet_layers", type=int, default=3)
    parser.add_argument("--pointnet_variation", type=int, default=0)
    parser.add_argument("--pointnet_numpoints", type=int, default=256)
    parser.add_argument("--pointnet_path", type=str, default="./checkpoints/pointnet_acc0.86_lr1_p256.pth")
    parser.add_argument("--pointnet_freeze", action="store_true")
    parser.add_argument("--pointnet_features", type=int, default=2)
    parser.add_argument("--class_embed", action="store_true")
    parser.add_argument("--color_embed", action="store_true")
    parser.add_argument("--object_size", type=int, default=28)
    parser.add_argument("--object_inter_module_num_heads", type=int, default=4)
    parser.add_argument("--object_inter_module_num_layers", type=int, default=2)
    
    # Language Encoder
    parser.add_argument("--hungging_model", type=str, help="hugging face model")
    parser.add_argument("--fixed_embedding", action="store_true")
    parser.add_argument("--inter_module_num_heads", type=int, default=4)
    parser.add_argument("--inter_module_num_layers", type=int, default=1)
    parser.add_argument("--intra_module_num_heads", type=int, default=4)
    parser.add_argument("--intra_module_num_layers", type=int, default=1)
    parser.add_argument("--fine_intra_module_num_heads", type=int, default=4)
    parser.add_argument("--fine_intra_module_num_layers", type=int, default=1)
    
    # Variations
    parser.add_argument("--regressor_cell", type=str, default="pose")
    parser.add_argument("--regressor_learn", type=str, default="center")
    parser.add_argument("--regressor_eval", type=str, default="center")
    parser.add_argument("--variation", type=int, default=0)
    
    # MSG
    parser.add_argument("--num_of_hidden_layer", type=int, default=3)
    parser.add_argument("--alpha", type=int, default=2)
    
    # Fine-localization
    parser.add_argument("--coarse_path_in_fine", type=str, default=None)
    parser.add_argument("--confidence_thresh", type=float, default=0.15)
    
    # Others
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr_gamma", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="exponential")
    parser.add_argument("--lr_step", type=float, default=10)
    parser.add_argument("--folder_name", type=str, default="folder_name")
    parser.add_argument("--cpus", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="adam")
    
    # Branch-specific arguments
    add_branch_arguments(parser)
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = EasyDict(vars(args))
    
    # Override with explicit args
    if args.epochs_per_branch is None:
        args.epochs_per_branch = args.epochs
    if args.lr is None:
        args.lr = args.learning_rate

    print(f"\n{'#'*70}")
    print(f"# Branch Training Configuration")
    print(f"{'#'*70}")
    print(f"# Branch: {args.branch}")
    print(f"# Epochs per branch: {args.epochs_per_branch}")
    print(f"# Learning rate: {args.lr}")
    print(f"# Save directory: ./checkpoints/")
    print(f"{'#'*70}\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}\n")

    # Create data loaders
    if args.no_pc_augment:
        train_transform = T.FixedPoints(args.pointnet_numpoints)
        val_transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        train_transform = T.Compose([
            T.FixedPoints(args.pointnet_numpoints),
            T.RandomRotate(120, axis=2),
            T.NormalizeScale(),
        ])
        val_transform = T.Compose([
            T.FixedPoints(args.pointnet_numpoints),
            T.NormalizeScale()
        ])

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
        args.base_path, SCENE_NAMES_VAL, val_transform,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    # Train branches
    if args.branch in ['all', 'global']:
        save_dir = create_save_dir('global')
        model = GlobalBranch(
            dataset_train.get_known_classes(),
            COLOR_NAMES_K360,
            args
        )
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        train_global_branch(
            model, dataloader_train, dataloader_val,
            optimizer, args.epochs_per_branch, args, save_dir
        )
        
        del model
        torch.cuda.empty_cache()

    if args.branch in ['all', 'object']:
        save_dir = create_save_dir('object')
        model = ObjectBranch(
            dataset_train.get_known_classes(),
            COLOR_NAMES_K360,
            args
        )
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        train_object_branch(
            model, dataloader_train, dataloader_val,
            optimizer, args.epochs_per_branch, args, save_dir
        )
        
        del model
        torch.cuda.empty_cache()

    if args.branch in ['all', 'relation']:
        save_dir = create_save_dir('relation')
        model = RelationBranch(
            dataset_train.get_known_classes(),
            COLOR_NAMES_K360,
            args
        )
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        train_relation_branch(
            model, dataloader_train, dataloader_val,
            optimizer, args.epochs_per_branch, args, save_dir
        )
        
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Checkpoints saved to:")
    if args.branch in ['all', 'global']:
        print(f"  - ./checkpoints/global/")
    if args.branch in ['all', 'object']:
        print(f"  - ./checkpoints/object/")
    if args.branch in ['all', 'relation']:
        print(f"  - ./checkpoints/relation/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
