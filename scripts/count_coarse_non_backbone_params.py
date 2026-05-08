"""Count coarse-stage parameters excluding backbone modules.

By default, this script treats the following as backbones and excludes them:
- ``object_encoder.pointnet``: point-cloud backbone
- ``language_encoder.llm_model``: HuggingFace text backbone

It builds the coarse model ``CellRetrievalNetwork`` with the same default
hyperparameters as training and reports:
- total parameters
- trainable parameters
- backbone parameters
- non-backbone parameters
- non-backbone trainable parameters

You can optionally override the excluded module prefixes.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from typing import Iterable

from datapreparation.kitti360pose.utils import COLOR_NAMES, KNOWN_CLASS
from models.cell_retrieval import CellRetrievalNetwork


DEFAULT_BACKBONE_PREFIXES = (
    "object_encoder.pointnet",
    "language_encoder.llm_model",
)


def build_model_args(cli_args: argparse.Namespace) -> Namespace:
    """Create a model-args namespace compatible with ``CellRetrievalNetwork``."""

    return Namespace(
        coarse_embed_dim=cli_args.coarse_embed_dim,
        pointnet_layers=cli_args.pointnet_layers,
        pointnet_variation=cli_args.pointnet_variation,
        pointnet_numpoints=cli_args.pointnet_numpoints,
        pointnet_path=cli_args.pointnet_path,
        pointnet_freeze=cli_args.pointnet_freeze,
        pointnet_features=cli_args.pointnet_features,
        class_embed=cli_args.class_embed,
        color_embed=cli_args.color_embed,
        object_size=cli_args.object_size,
        object_inter_module_num_heads=cli_args.object_inter_module_num_heads,
        object_inter_module_num_layers=cli_args.object_inter_module_num_layers,
        hungging_model=cli_args.hungging_model,
        fixed_embedding=cli_args.fixed_embedding,
        inter_module_num_heads=cli_args.inter_module_num_heads,
        inter_module_num_layers=cli_args.inter_module_num_layers,
        intra_module_num_heads=cli_args.intra_module_num_heads,
        intra_module_num_layers=cli_args.intra_module_num_layers,
        num_of_hidden_layer=cli_args.num_of_hidden_layer,
        use_features=cli_args.use_features,
    )


def count_parameters(model, excluded_prefixes: Iterable[str]) -> dict[str, int]:
    """Count model parameters with and without excluded backbone modules."""

    excluded_prefixes = tuple(excluded_prefixes)
    total = 0
    total_trainable = 0
    backbone = 0
    backbone_trainable = 0
    non_backbone = 0
    non_backbone_trainable = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        is_backbone = any(name.startswith(prefix) for prefix in excluded_prefixes)

        total += numel
        if param.requires_grad:
            total_trainable += numel

        if is_backbone:
            backbone += numel
            if param.requires_grad:
                backbone_trainable += numel
        else:
            non_backbone += numel
            if param.requires_grad:
                non_backbone_trainable += numel

    return {
        "total": total,
        "total_trainable": total_trainable,
        "backbone": backbone,
        "backbone_trainable": backbone_trainable,
        "non_backbone": non_backbone,
        "non_backbone_trainable": non_backbone_trainable,
    }


def summarize_by_top_module(model, excluded_prefixes: Iterable[str]) -> list[tuple[str, int, bool]]:
    """Aggregate parameter counts by top-level child module."""

    excluded_prefixes = tuple(excluded_prefixes)
    summary: dict[str, int] = {}
    flags: dict[str, bool] = {}

    for name, param in model.named_parameters():
        top_module = name.split(".", 1)[0]
        summary[top_module] = summary.get(top_module, 0) + param.numel()
        flags[top_module] = flags.get(top_module, False) or any(
            name.startswith(prefix) for prefix in excluded_prefixes
        )

    return sorted(
        [(module, count, flags[module]) for module, count in summary.items()],
        key=lambda x: x[1],
        reverse=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count coarse-stage parameters excluding backbone modules."
    )

    parser.add_argument(
        "--exclude_prefixes",
        nargs="+",
        default=list(DEFAULT_BACKBONE_PREFIXES),
        help="Parameter-name prefixes to exclude as backbone modules.",
    )

    parser.add_argument("--coarse_embed_dim", type=int, default=256)
    parser.add_argument("--pointnet_layers", type=int, default=3)
    parser.add_argument("--pointnet_variation", type=int, default=0)
    parser.add_argument("--pointnet_numpoints", type=int, default=256)
    parser.add_argument(
        "--pointnet_path",
        type=str,
        default="./checkpoints/pointnet_acc0.86_lr1_p256.pth",
    )
    parser.add_argument("--pointnet_freeze", action="store_true")
    parser.add_argument("--pointnet_features", type=int, default=2)
    parser.add_argument("--class_embed", action="store_true")
    parser.add_argument("--color_embed", action="store_true")
    parser.add_argument("--object_size", type=int, default=28)
    parser.add_argument("--object_inter_module_num_heads", type=int, default=4)
    parser.add_argument("--object_inter_module_num_layers", type=int, default=2)
    parser.add_argument("--hungging_model", type=str, required=True)
    parser.add_argument("--fixed_embedding", action="store_true")
    parser.add_argument("--inter_module_num_heads", type=int, default=4)
    parser.add_argument("--inter_module_num_layers", type=int, default=1)
    parser.add_argument("--intra_module_num_heads", type=int, default=4)
    parser.add_argument("--intra_module_num_layers", type=int, default=1)
    parser.add_argument("--num_of_hidden_layer", type=int, default=3)
    parser.add_argument(
        "--use_features",
        nargs="+",
        default=["class", "color", "position", "num"],
    )

    return parser.parse_args()


def format_count(num: int) -> str:
    return f"{num:,}"


def main() -> int:
    cli_args = parse_args()
    model_args = build_model_args(cli_args)

    model = CellRetrievalNetwork(KNOWN_CLASS, COLOR_NAMES, model_args)
    counts = count_parameters(model, cli_args.exclude_prefixes)
    module_summary = summarize_by_top_module(model, cli_args.exclude_prefixes)

    print("Excluded backbone prefixes:")
    for prefix in cli_args.exclude_prefixes:
        print(f"  - {prefix}")

    print("\nParameter counts:")
    print(f"  total params                : {format_count(counts['total'])}")
    print(f"  total trainable params      : {format_count(counts['total_trainable'])}")
    print(f"  backbone params             : {format_count(counts['backbone'])}")
    print(f"  backbone trainable params   : {format_count(counts['backbone_trainable'])}")
    print(f"  non-backbone params         : {format_count(counts['non_backbone'])}")
    print(f"  non-backbone trainable      : {format_count(counts['non_backbone_trainable'])}")

    print("\nTop-level module breakdown:")
    for module, count, has_backbone in module_summary:
        tag = "contains excluded backbone params" if has_backbone else "non-backbone"
        print(f"  - {module:<20} {format_count(count):>15}   {tag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
