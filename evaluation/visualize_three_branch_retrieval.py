"""Visualization utilities for three-branch retrieval results.

This module supports two workflows:
1. Run inference and save retrieval results to a JSON file.
2. Load saved retrieval results and render BEV visualizations.

The visual layout is designed for each query to show:
- query text
- a 10x10m BEV around the query GT position
- top-5 coarse retrieved candidate regions
- fine-predicted coordinates marked on every candidate panel

The implementation is intentionally framework-light so it can be unit-tested
without depending on the full training pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RetrievalCandidate:
    """One coarse candidate with optional fine prediction."""

    cell_id: str
    center_xy: Tuple[float, float]
    coarse_score: float
    fine_xy: Optional[Tuple[float, float]] = None
    fine_score: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalCandidate":
        return cls(
            cell_id=str(data["cell_id"]),
            center_xy=tuple(data["center_xy"]),
            coarse_score=float(data.get("coarse_score", 0.0)),
            fine_xy=tuple(data["fine_xy"]) if data.get("fine_xy") is not None else None,
            fine_score=float(data["fine_score"]) if data.get("fine_score") is not None else None,
        )


@dataclass
class RetrievalRecord:
    """Visualization-ready retrieval record for a single query."""

    query_id: str
    query_text: str
    gt_xy: Tuple[float, float]
    query_points_xy: List[Tuple[float, float]]
    candidates: List[RetrievalCandidate]
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalRecord":
        return cls(
            query_id=str(data["query_id"]),
            query_text=str(data["query_text"]),
            gt_xy=tuple(data["gt_xy"]),
            query_points_xy=[tuple(p) for p in data.get("query_points_xy", [])],
            candidates=[RetrievalCandidate.from_dict(c) for c in data.get("candidates", [])],
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def load_records(path: str | Path) -> List[RetrievalRecord]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    return [RetrievalRecord.from_dict(item) for item in records]


def save_records(records: Sequence[RetrievalRecord | Dict[str, Any]], path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [r.to_dict() if hasattr(r, "to_dict") else r for r in records]
    payload = {
        "metadata": metadata or {},
        "records": normalized,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def crop_points(points_xy: np.ndarray, center_xy: Tuple[float, float], size_m: float = 10.0) -> np.ndarray:
    """Return points inside an axis-aligned square centered at `center_xy`."""

    points_xy = np.asarray(points_xy, dtype=np.float32)
    if points_xy.size == 0:
        return points_xy.reshape(0, 2)

    half = size_m / 2.0
    cx, cy = center_xy
    mask = (
        (points_xy[:, 0] >= cx - half)
        & (points_xy[:, 0] <= cx + half)
        & (points_xy[:, 1] >= cy - half)
        & (points_xy[:, 1] <= cy + half)
    )
    return points_xy[mask]


def _scatter_bev(ax: plt.Axes, points_xy: np.ndarray, center_xy: Tuple[float, float], size_m: float = 10.0) -> None:
    ax.set_aspect("equal", adjustable="box")
    half = size_m / 2.0
    cx, cy = center_xy
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.grid(True, linewidth=0.3, alpha=0.25)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    if points_xy.size > 0:
        ax.scatter(points_xy[:, 0], points_xy[:, 1], s=2, c="#666666", alpha=0.55, linewidths=0)


def plot_single_query(record: RetrievalRecord, bev_size_m: float = 10.0, topk: int = 5, show_gt: bool = True):
    """Create one row visualization for a single query."""

    candidates = list(record.candidates[:topk])
    n_panels = 2 + len(candidates)
    fig_w = max(14, 3.2 * n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 3.8), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    query_points = np.asarray(record.query_points_xy, dtype=np.float32)

    # Query text panel.
    ax0 = axes[0]
    ax0.axis("off")
    meta = [f"Query ID: {record.query_id}", f"GT: ({record.gt_xy[0]:.2f}, {record.gt_xy[1]:.2f})"]
    text = "\n".join(meta + ["", record.query_text])
    ax0.text(0.0, 0.95, text, va="top", ha="left", fontsize=10, wrap=True)
    ax0.set_title("Query Text", fontsize=11)

    # Query BEV panel.
    ax1 = axes[1]
    _scatter_bev(ax1, crop_points(query_points, record.gt_xy, bev_size_m), record.gt_xy, bev_size_m)
    if show_gt:
        ax1.scatter([record.gt_xy[0]], [record.gt_xy[1]], c="limegreen", s=60, marker="*", edgecolors="black", linewidths=0.4)
    ax1.set_title("Query GT BEV", fontsize=11)

    # Candidate panels.
    for idx, candidate in enumerate(candidates, start=2):
        ax = axes[idx]
        candidate_points = crop_points(query_points, candidate.center_xy, bev_size_m)
        _scatter_bev(ax, candidate_points, candidate.center_xy, bev_size_m)
        ax.scatter([candidate.center_xy[0]], [candidate.center_xy[1]], c="dodgerblue", s=55, marker="x", linewidths=1.6)
        if candidate.fine_xy is not None:
            ax.scatter([candidate.fine_xy[0]], [candidate.fine_xy[1]], c="crimson", s=55, marker="o", edgecolors="white", linewidths=0.6)
        if show_gt:
            ax.scatter([record.gt_xy[0]], [record.gt_xy[1]], c="limegreen", s=45, marker="*", edgecolors="black", linewidths=0.4)
        title = [f"Top-{idx-1} | {candidate.cell_id}", f"coarse={candidate.coarse_score:.4f}"]
        if candidate.fine_score is not None:
            title.append(f"fine={candidate.fine_score:.4f}")
        ax.set_title("\n".join(title), fontsize=9)

    return fig


def render_records(records: Sequence[RetrievalRecord], out_dir: str | Path, bev_size_m: float = 10.0, topk: int = 5) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []
    for record in records:
        fig = plot_single_query(record, bev_size_m=bev_size_m, topk=topk)
        out_path = out_dir / f"{record.query_id}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def build_demo_records(num_records: int = 2, topk: int = 5) -> List[RetrievalRecord]:
    records: List[RetrievalRecord] = []
    rng = np.random.default_rng(0)
    for i in range(num_records):
        gt = (float(i * 10.0), float(i * 5.0))
        points = gt + rng.normal(scale=1.5, size=(400, 2))
        candidates = []
        for k in range(topk):
            center = (gt[0] + (k - 2) * 1.2, gt[1] + (k - 2) * 0.8)
            candidates.append(
                RetrievalCandidate(
                    cell_id=f"cell_{i}_{k}",
                    center_xy=center,
                    coarse_score=1.0 / (k + 1),
                    fine_xy=(center[0] + 0.3, center[1] - 0.2),
                    fine_score=0.9 / (k + 1),
                )
            )
        records.append(
            RetrievalRecord(
                query_id=f"query_{i}",
                query_text=f"This is a demo query {i}.",
                gt_xy=gt,
                query_points_xy=[tuple(p) for p in points],
                candidates=candidates,
            )
        )
    return records


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize three-branch retrieval results.")
    parser.add_argument("--input", type=str, help="Path to saved retrieval JSON file.")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Directory to write PNG files.")
    parser.add_argument("--save_results", type=str, help="Optional path to save demo or runtime retrieval results.")
    parser.add_argument("--demo", action="store_true", help="Generate demo records instead of loading an input file.")
    parser.add_argument("--num_records", type=int, default=2)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--bev_size", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.demo:
        records = build_demo_records(num_records=args.num_records, topk=args.topk)
        if args.save_results:
            save_records(records, args.save_results, metadata={"source": "demo"})
    else:
        if not args.input:
            raise SystemExit("--input is required unless --demo is set")
        records = load_records(args.input)

    render_records(records, args.output_dir, bev_size_m=args.bev_size, topk=args.topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
