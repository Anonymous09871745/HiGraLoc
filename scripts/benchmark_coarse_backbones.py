"""Benchmark coarse-stage backbone inference speed.

This script measures the forward latency of:
1. one text through the T5 encoder backbone
2. one point cloud through the PointNet++ backbone

It intentionally benchmarks only the two backbone modules, not the extra coarse
fusion/MSG/Transformer heads.
"""

from __future__ import annotations

import argparse
import time
from argparse import Namespace
from statistics import mean, stdev

import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, T5EncoderModel

from datapreparation.kitti360pose.utils import COLOR_NAMES, KNOWN_CLASS
from models.pointcloud.pointnet2 import PointNet2


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


def benchmark_t5(args: argparse.Namespace, device: torch.device) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.hugging_model,
        local_files_only=args.local_files_only,
    )
    model = T5EncoderModel.from_pretrained(
        args.hugging_model,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    encoded = tokenizer(
        [args.text] * args.text_batch_size,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=args.max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    def forward_once():
        return model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_attentions=False,
        ).last_hidden_state

    return benchmark_callable(forward_once, args.warmup, args.runs, device)


def build_pointnet_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        pointnet_layers=args.pointnet_layers,
        pointnet_variation=args.pointnet_variation,
    )


def make_random_point_cloud(args: argparse.Namespace, device: torch.device) -> Data:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    pos = torch.rand((args.num_points, 3), generator=generator, dtype=torch.float32)
    x = torch.rand((args.num_points, 3), generator=generator, dtype=torch.float32)
    batch = torch.zeros(args.num_points, dtype=torch.long)
    return Data(x=x.to(device), pos=pos.to(device), batch=batch.to(device))


def benchmark_pointnet(args: argparse.Namespace, device: torch.device) -> list[float]:
    pointnet = PointNet2(
        num_classes=len(KNOWN_CLASS),
        num_colors=len(COLOR_NAMES),
        args=build_pointnet_args(args),
    ).to(device)

    if args.pointnet_checkpoint:
        state = torch.load(args.pointnet_checkpoint, map_location=device)
        pointnet.load_state_dict(state)

    pointnet.eval()
    point_cloud = make_random_point_cloud(args, device)

    def forward_once():
        return pointnet(point_cloud).features2

    return benchmark_callable(forward_once, args.warmup, args.runs, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one-text T5 and one-point-cloud PointNet++ inference speed.")
    parser.add_argument("--hugging_model", required=True, help="T5 model name or local path.")
    parser.add_argument("--local_files_only", action="store_true", help="Do not access HuggingFace network.")
    parser.add_argument(
        "--text",
        default="Pose is on-top of a gray road. Pose is north of a green sidewalk.",
        help="Input text for T5 speed test.",
    )
    parser.add_argument("--text_batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_points", type=int, default=256)
    parser.add_argument("--pointnet_checkpoint", default="./checkpoints/pointnet_acc0.86_lr1_p256.pth")
    parser.add_argument("--pointnet_layers", type=int, default=3)
    parser.add_argument("--pointnet_variation", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_t5", action="store_true")
    parser.add_argument("--skip_pointnet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Warmup: {args.warmup}, runs: {args.runs}")

    if not args.skip_t5:
        t5_times = benchmark_t5(args, device)
        format_stats("T5 encoder backbone, one text", t5_times, batch_size=args.text_batch_size)

    if not args.skip_pointnet:
        pointnet_times = benchmark_pointnet(args, device)
        format_stats(f"PointNet++ backbone, one point cloud ({args.num_points} points)", pointnet_times)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
