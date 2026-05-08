"""Render 6 paper-style retrieval rows whose top-1/top-2/top-3 are all correct.

This script is a filtered variant of `render_paper_retrieval_figure_grid.py`.
It automatically scans the retrieval JSON, selects records where the first
three retrieved candidates are correct, then renders a 6-row figure and exports
the corresponding original text queries.

The correctness criterion is intentionally the same as the green/red outer-box
logic used by the renderer: a candidate is correct if the distance between the
GT query position and the candidate cell center is <= `--correct_thresh` meters.
With the default threshold of 10m, top1, top2, and top3 will all be rendered with
green outer boxes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from evaluation.render_paper_retrieval_figure_grid import render_six_query_grid
from evaluation.visualize_three_branch_retrieval import load_records


def _candidate_is_green(record, candidate, correct_thresh: float) -> bool:
    """Return True if this candidate will be drawn with a green outer box."""

    distance = float(np.linalg.norm(np.asarray(record.gt_xy) - np.asarray(candidate.center_xy)))
    return distance <= correct_thresh


def _top123_are_green(record, correct_thresh: float) -> bool:
    """Return True only when top-1, top-2, and top-3 are all green/correct."""

    if len(record.candidates) < 3:
        return False
    return all(_candidate_is_green(record, candidate, correct_thresh) for candidate in record.candidates[:3])


def find_top123_correct_indices(retrieval_json: str, num_rows: int = 6, correct_thresh: float = 10.0, start_idx: int = 0) -> list[int]:
    """Find query indices whose top1/top2/top3 candidates are all correct."""

    records = load_records(retrieval_json)
    selected_indices = [
        idx
        for idx, record in enumerate(records)
        if idx >= start_idx and _top123_are_green(record, correct_thresh)
    ]

    if len(selected_indices) < num_rows:
        raise ValueError(
            f"Only found {len(selected_indices)} records where top1/top2/top3 are all correct "
            f"with threshold <= {correct_thresh:.2f}m, but {num_rows} rows were requested."
        )

    return selected_indices[:num_rows]


def write_selected_indices_json(
    selected_indices_json: str | None,
    retrieval_json: str,
    selected_indices: Sequence[int],
    correct_thresh: float,
) -> None:
    """Optionally save the selected query indices for reproducibility."""

    if selected_indices_json is None:
        return

    records = load_records(retrieval_json)
    payload = {
        "retrieval_json": str(retrieval_json),
        "correct_thresh": float(correct_thresh),
        "selected_indices": list(selected_indices),
        "selected": [
            {
                "query_idx": int(idx),
                "query_id": str(records[idx].query_id),
                "gt_xy": [float(records[idx].gt_xy[0]), float(records[idx].gt_xy[1])],
                "top123_center_distances": [
                    float(np.linalg.norm(np.asarray(records[idx].gt_xy) - np.asarray(candidate.center_xy)))
                    for candidate in records[idx].candidates[:3]
                ],
                "query_text": str(records[idx].query_text),
            }
            for idx in selected_indices
        ],
    }

    path = Path(selected_indices_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 6 rows where top1/top2/top3 are all green/correct.")
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--retrieval_json", required=True)
    parser.add_argument("--output", required=True, help="Output multi-row PNG path. A per-run folder is still created automatically.")
    parser.add_argument("--query_json", required=True, help="Output JSON path for the six original text queries. A per-run folder is still created automatically.")
    parser.add_argument("--num_rows", type=int, default=6, help="Number of rows to render. Defaults to 6.")
    parser.add_argument("--correct_thresh", type=float, default=10.0, help="Distance threshold in meters for green/correct candidates.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start scanning records from this query index.")
    parser.add_argument("--bev_size", type=float, default=50.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--row_gap", type=int, default=90)
    parser.add_argument("--selected_indices_json", default=None, help="Optional path to save the automatically selected query indices.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    selected_indices = find_top123_correct_indices(
        args.retrieval_json,
        num_rows=args.num_rows,
        correct_thresh=args.correct_thresh,
        start_idx=args.start_idx,
    )

    write_selected_indices_json(
        args.selected_indices_json,
        args.retrieval_json,
        selected_indices,
        args.correct_thresh,
    )

    render_six_query_grid(
        args.base_path,
        args.scene,
        args.retrieval_json,
        args.output,
        args.query_json,
        selected_indices,
        bev_size=args.bev_size,
        topk=args.topk,
        row_gap=args.row_gap,
    )
    print("Selected query indices:", " ".join(str(idx) for idx in selected_indices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
