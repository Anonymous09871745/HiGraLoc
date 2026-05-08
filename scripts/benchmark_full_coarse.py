"""Benchmark full coarse-stage forward latency.

This script measures the end-to-end coarse forward path used by the three-branch
pipeline:
- T5 text encoding for Global / Object / Relation branches
- PointNet++ object encoding for Global / Object / Relation branches
- MSG / ConvBlock / branch heads inside each branch
- weighted coarse score fusion across the three branches

By default it benchmarks one batch from the retrieval dataset and one batch from
its cell dataset, then times the complete coarse forward pass on that batch.
"""

from __future__ import annotations

import argparse
import time
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
import torch_geometric.transforms as T
from torch.utils.data import DataLoader

from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360, KNOWN_CLASS, SCENE_NAMES_TEST, SCENE_NAMES_VAL
from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from evaluation.pipeline_three_branch_with_fine import compute_weighted_scores
from models.branches import GlobalBranch, ObjectBranch, RelationBranch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    values_sorted = sorted(values)
    idx = min(len(values_sorted) - 1, max(0, round((len(values_sorted) - 1) * q)))
    return values_sorted[idx]


def format_stats(name: str, times_ms: list[float]) -> None:
    avg = mean(times_ms)
    sd = stdev(times_ms) if len(times_ms) > 1 else 0.0
    print(f"\n{name}")
    print(f"  runs              : {len(times_ms)}")
    print(f"  mean latency      : {avg:.3f} ms")
    print(f"  std latency       : {sd:.3f} ms")
    print(f"  p50 latency       : {percentile(times_ms, 0.50):.3f} ms")
    print(f"  p90 latency       : {percentile(times_ms, 0.90):.3f} ms")
    print(f"  p95 latency       : {percentile(times_ms, 0.95):.3f} ms")
    print(f"  throughput        : {1000.0 / avg:.2f} queries/s")


def build_transform(args: argparse.Namespace):
    if args.no_pc_augment:
        return T.FixedPoints(args.pointnet_numpoints)
    return T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])


def build_dataset(args: argparse.Namespace, transform):
    scenes = SCENE_NAMES_TEST if args.use_test_set else SCENE_NAMES_VAL
    dataset = Kitti360CoarseDatasetMulti(
        args.base_path,
        scenes,
        transform,
        shuffle_hints=False,
        flip_poses=False,
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
    return dataset, dataloader, cells_dataset, cells_dataloader


def load_model(cls, args: argparse.Namespace, checkpoint: str, device: torch.device):
    model = cls(KNOWN_CLASS, COLOR_NAMES_K360, args)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def benchmark_full_coarse(args: argparse.Namespace, device: torch.device) -> list[float]:
    transform = build_transform(args)
    dataset, dataloader, cells_dataset, cells_dataloader = build_dataset(args, transform)

    model_global = load_model(GlobalBranch, args, args.global_checkpoint, device)
    model_object = load_model(ObjectBranch, args, args.object_checkpoint, device)
    model_relation = load_model(RelationBranch, args, args.relation_checkpoint, device)

    batch_queries = next(iter(dataloader))
    batch_cells = next(iter(cells_dataloader))

    def forward_once() -> Any:
        encodings = {
            "query_cell_ids": np.array(batch_queries["cell_ids"]),
            "db_cell_ids": np.array(batch_cells["cell_ids"]),
        }

        # Query encodings
        encodings["text_enc_global"] = model_global.encode_text(batch_queries["texts"]).detach().cpu().numpy()
        encodings["text_enc_object"] = model_object.encode_text(batch_queries["texts"]).detach().cpu().numpy()
        encodings["text_enc_relation"] = model_relation.encode_text(batch_queries["texts"]).detach().cpu().numpy()

        # Cell encodings
        encodings["cell_enc_global"] = model_global.encode_objects(batch_cells["objects"], batch_cells["object_points"]).detach().cpu().numpy()
        obj_cell, obj_mask = model_object.encode_objects(batch_cells["objects"], batch_cells["object_points"])
        rel_cell, rel_mask = model_relation.encode_objects(batch_cells["objects"], batch_cells["object_points"])

        encodings["cell_enc_object"] = obj_cell.detach().cpu().numpy()
        encodings["cell_mask_object"] = obj_mask.detach().cpu().numpy()
        encodings["cell_enc_relation"] = rel_cell.detach().cpu().numpy()
        encodings["cell_mask_relation"] = rel_mask.detach().cpu().numpy()

        # Fusion / score computation
        return compute_weighted_scores(encodings, args, logger=None)

    with torch.inference_mode():
        for _ in range(args.warmup):
            forward_once()
        synchronize(device)

        times_ms: list[float] = []
        for _ in range(args.runs):
            synchronize(device)
            start = time.perf_counter()
            forward_once()
            synchronize(device)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    return times_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full coarse-stage latency.")
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--use_test_set", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--coarse_embed_dim", type=int, default=256)
    parser.add_argument("--pointnet_layers", type=int, default=3)
    parser.add_argument("--pointnet_variation", type=int, default=0)
    parser.add_argument("--pointnet_numpoints", type=int, default=256)
    parser.add_argument("--pointnet_path", default="./checkpoints/pointnet_acc0.86_lr1_p256.pth")
    parser.add_argument("--pointnet_freeze", action="store_true")
    parser.add_argument("--pointnet_features", type=int, default=2)
    parser.add_argument("--class_embed", action="store_true")
    parser.add_argument("--color_embed", action="store_true")
    parser.add_argument("--object_size", type=int, default=28)
    parser.add_argument("--object_inter_module_num_heads", type=int, default=4)
    parser.add_argument("--object_inter_module_num_layers", type=int, default=2)
    parser.add_argument("--hungging_model", required=True)
    parser.add_argument("--fixed_embedding", action="store_true")
    parser.add_argument("--inter_module_num_heads", type=int, default=4)
    parser.add_argument("--inter_module_num_layers", type=int, default=1)
    parser.add_argument("--intra_module_num_heads", type=int, default=4)
    parser.add_argument("--intra_module_num_layers", type=int, default=1)
    parser.add_argument("--num_of_hidden_layer", type=int, default=3)
    parser.add_argument("--use_features", nargs="+", default=["class", "color", "position", "num"])
    parser.add_argument("--weight_global", type=float, default=0.5)
    parser.add_argument("--weight_object", type=float, default=1.5)
    parser.add_argument("--weight_relation", type=float, default=0.5)
    parser.add_argument("--global_checkpoint", required=True)
    parser.add_argument("--object_checkpoint", required=True)
    parser.add_argument("--relation_checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_pc_augment", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Warmup: {args.warmup}, runs: {args.runs}")

    times = benchmark_full_coarse(args, device)
    format_stats("Full coarse forward (T5 + PointNet++ + fusion/MSG/Transformer)", times)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
