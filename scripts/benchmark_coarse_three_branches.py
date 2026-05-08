"""Benchmark the full coarse-stage speed for three branches with shared backbone.

This script measures the end-to-end coarse inference cost for:
- one text query encoded by a shared T5 backbone
- one point-cloud cell encoded by a shared PointNet++ backbone
- the three coarse branches using the shared backbone features

It reports both:
1. backbone-only cost, and
2. full three-branch coarse cost

The shared-backbone measurement is approximated by loading the shared backbones
once and reusing them across the three branches, which matches the intended
coarse-stage setup.
"""

from __future__ import annotations

import argparse
import time
from argparse import Namespace
from statistics import mean, stdev

import numpy as np
import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, T5EncoderModel

from datapreparation.kitti360pose.utils import COLOR_NAMES, KNOWN_CLASS
from models.bir_lstm import ConvBlock
from models.language_encoder import LanguageEncoder
from models.msg_encoder import ObjectMsgEncoder
from models.object_encoder import ObjectEncoder
from models.branches.global_branch import GlobalBranch
from models.branches.object_branch import ObjectBranch
from models.branches.relation_branch import RelationBranch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = min(len(values_sorted) - 1, max(0, round((len(values_sorted) - 1) * q)))
    return values_sorted[idx]


def format_stats(name: str, times_ms: list[float], batch_size: int = 1) -> None:
    avg = mean(times_ms)
    sd = stdev(times_ms) if len(times_ms) > 1 else 0.0
    print(f"\n{name}")
    print(f"  runs              : {len(times_ms)}")
    print(f"  mean latency      : {avg:.3f} ms")
    print(f"  std latency       : {sd:.3f} ms")
    print(f"  p50 latency       : {percentile(times_ms, 0.50):.3f} ms")
    print(f"  p90 latency       : {percentile(times_ms, 0.90):.3f} ms")
    print(f"  p95 latency       : {percentile(times_ms, 0.95):.3f} ms")
    print(f"  throughput        : {1000.0 * batch_size / avg:.2f} samples/s")


def benchmark_callable(fn, warmup: int, runs: int, device: torch.device) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        synchronize(device)

        times_ms: list[float] = []
        for _ in range(runs):
            synchronize(device)
            start = time.perf_counter()
            fn()
            synchronize(device)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    return times_ms


def build_args(cli_args: argparse.Namespace) -> Namespace:
    return Namespace(
        coarse_embed_dim=cli_args.coarse_embed_dim,
        pointnet_layers=cli_args.pointnet_layers,
        pointnet_variation=cli_args.pointnet_variation,
        pointnet_numpoints=cli_args.pointnet_numpoints,
        pointnet_path=cli_args.pointnet_checkpoint,
        pointnet_freeze=cli_args.pointnet_freeze,
        pointnet_features=cli_args.pointnet_features,
        class_embed=cli_args.class_embed,
        color_embed=cli_args.color_embed,
        object_size=cli_args.object_size,
        object_inter_module_num_heads=cli_args.object_inter_module_num_heads,
        object_inter_module_num_layers=cli_args.object_inter_module_num_layers,
        hungging_model=cli_args.hugging_model,
        fixed_embedding=cli_args.fixed_embedding,
        inter_module_num_heads=cli_args.inter_module_num_heads,
        inter_module_num_layers=cli_args.inter_module_num_layers,
        intra_module_num_heads=cli_args.intra_module_num_heads,
        intra_module_num_layers=cli_args.intra_module_num_layers,
        num_of_hidden_layer=cli_args.num_of_hidden_layer,
        use_features=cli_args.use_features,
        num_mentioned=cli_args.num_mentioned,
        no_isre=cli_args.no_isre,
        use_riemannian=cli_args.use_riemannian,
        riemannian_manifold=cli_args.riemannian_manifold,
    )


def make_point_cloud(num_points: int, device: torch.device, seed: int = 0) -> Data:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pos = torch.rand((num_points, 3), generator=generator, dtype=torch.float32)
    x = torch.rand((num_points, 3), generator=generator, dtype=torch.float32)
    batch = torch.zeros(num_points, dtype=torch.long)
    return Data(x=x.to(device), pos=pos.to(device), batch=batch.to(device))


def make_objects_batch(num_objects: int, args: argparse.Namespace):
    objects = []
    points = []
    for i in range(num_objects):
        cls = KNOWN_CLASS[i % len(KNOWN_CLASS)]
        color = COLOR_NAMES[i % len(COLOR_NAMES)]
        obj = type("DummyObj", (), {})()
        obj.label = cls
        obj.get_color_text = lambda c=color: c
        obj.get_center = lambda idx=i: np.array([float(idx), float(idx) * 0.1, float(idx) * 0.01], dtype=np.float32)
        obj.get_color_rgb = lambda idx=i: np.array([idx % 255, (idx * 2) % 255, (idx * 3) % 255], dtype=np.float32)
        obj.xyz = np.random.rand(args.pointnet_numpoints, 3).astype(np.float32)
        objects.append(obj)

        # simple PyG-like object point cloud wrapper is not needed for timing if
        # object_encoder is skipped; however the full branches require it, so we
        # create a minimal Data object.
        points.append(
            Data(
                x=torch.rand((args.pointnet_numpoints, 3), dtype=torch.float32),
                pos=torch.rand((args.pointnet_numpoints, 3), dtype=torch.float32),
                batch=torch.zeros(args.pointnet_numpoints, dtype=torch.long),
            )
        )
    return [objects], points


def benchmark_shared_backbone(args: argparse.Namespace, device: torch.device) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(args.hugging_model, local_files_only=args.local_files_only)
    t5 = T5EncoderModel.from_pretrained(args.hugging_model, local_files_only=args.local_files_only).to(device).eval()
    pointnet = ObjectEncoder(args.coarse_embed_dim, KNOWN_CLASS, COLOR_NAMES, args).to(device).eval().pointnet

    text = [args.text]
    encoded = tokenizer(text, return_tensors="pt", padding="longest", truncation=True, max_length=args.max_length)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    point_cloud = make_point_cloud(args.num_points, device, seed=args.seed)

    def forward_once():
        text_hidden = t5(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], output_attentions=False).last_hidden_state
        _ = text_hidden[:, 0]
        pn_out = pointnet(point_cloud).features2
        _ = pn_out

    return benchmark_callable(forward_once, args.warmup, args.runs, device)


def benchmark_full_three_branch(args: argparse.Namespace, device: torch.device) -> list[float]:
    model_args = build_args(args)
    global_branch = GlobalBranch(KNOWN_CLASS, COLOR_NAMES, model_args).to(device).eval()
    object_branch = ObjectBranch(KNOWN_CLASS, COLOR_NAMES, model_args).to(device).eval()
    relation_branch = RelationBranch(KNOWN_CLASS, COLOR_NAMES, model_args).to(device).eval()

    # Share the backbone modules across branches explicitly.
    shared_lang = global_branch.language_encoder
    shared_obj = global_branch.object_encoder
    object_branch.language_encoder = shared_lang
    relation_branch.language_encoder = shared_lang
    object_branch.object_encoder = shared_obj
    relation_branch.object_encoder = shared_obj

    # PointNet weights should already be loaded inside ObjectEncoder during init.
    texts = [args.text]
    objects, object_points = make_objects_batch(args.num_objects, args)

    def forward_once():
        g_text = global_branch.encode_text(texts)
        o_text = object_branch.encode_text(texts)
        r_text = relation_branch.encode_text(texts)

        g_obj = global_branch.encode_objects(objects, object_points)
        o_obj, _ = object_branch.encode_objects(objects, object_points)
        r_obj, _ = relation_branch.encode_objects(objects, object_points)

        _ = (g_text, o_text, r_text, g_obj, o_obj, r_obj)

    return benchmark_callable(forward_once, args.warmup, args.runs, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full coarse-stage speed with shared backbone and three branches.")
    parser.add_argument("--hugging_model", required=True, help="T5 model name or local path.")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--text", default="Pose is on-top of a gray road. Pose is north of a green sidewalk. Pose is west of a black pole. Pose is east of a beige traffic sign. Pose is on-top of a dark-green sidewalk. Pose is south of a bright-gray sidewalk.")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_points", type=int, default=256)
    parser.add_argument("--num_objects", type=int, default=28)
    parser.add_argument("--pointnet_checkpoint", default="./checkpoints/pointnet_acc0.86_lr1_p256.pth")
    parser.add_argument("--pointnet_layers", type=int, default=3)
    parser.add_argument("--pointnet_variation", type=int, default=0)
    parser.add_argument("--pointnet_numpoints", type=int, default=256)
    parser.add_argument("--pointnet_features", type=int, default=2)
    parser.add_argument("--pointnet_freeze", action="store_true")
    parser.add_argument("--coarse_embed_dim", type=int, default=256)
    parser.add_argument("--object_size", type=int, default=28)
    parser.add_argument("--object_inter_module_num_heads", type=int, default=4)
    parser.add_argument("--object_inter_module_num_layers", type=int, default=1)
    parser.add_argument("--inter_module_num_heads", type=int, default=4)
    parser.add_argument("--inter_module_num_layers", type=int, default=1)
    parser.add_argument("--intra_module_num_heads", type=int, default=4)
    parser.add_argument("--intra_module_num_layers", type=int, default=1)
    parser.add_argument("--num_of_hidden_layer", type=int, default=3)
    parser.add_argument("--use_features", nargs="+", default=["class", "color", "position", "num"])
    parser.add_argument("--class_embed", action="store_true")
    parser.add_argument("--color_embed", action="store_true")
    parser.add_argument("--fixed_embedding", action="store_true")
    parser.add_argument("--num_mentioned", type=int, default=6)
    parser.add_argument("--no_isre", action="store_true")
    parser.add_argument("--use_riemannian", action="store_true")
    parser.add_argument("--riemannian_manifold", default="hyperbolic")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_shared_backbone", action="store_true")
    parser.add_argument("--skip_full_three_branch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Warmup: {args.warmup}, runs: {args.runs}")

    if not args.skip_shared_backbone:
        shared_times = benchmark_shared_backbone(args, device)
        format_stats("Shared backbone only: one text T5 + one point cloud PointNet++", shared_times)

    if not args.skip_full_three_branch:
        full_times = benchmark_full_three_branch(args, device)
        format_stats("Full coarse stage: three branches with shared backbone", full_times)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
