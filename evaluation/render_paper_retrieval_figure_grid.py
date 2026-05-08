"""Render a 6-row paper-style retrieval visualization for KITTI360Pose.

Each row contains the same content as `render_paper_retrieval_figure.py`:
- left: ground-truth query BEV rendered from the original dataset cell
- right: top-k retrieved candidate cells with GT/predicted markers and distance

In addition to the image, this script writes a JSON file containing the original
text queries used for the rendered rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from evaluation.render_paper_retrieval_figure import (
    _add_distance_box,
    _crop_and_resize_panel,
    _dataset_object_points,
    _find_query_cell,
    _load_multiscene_dataset,
    _render_dataset_cell_panel,
)
from evaluation.render_query_bev import render_bev_from_points
from evaluation.visualize_three_branch_retrieval import load_records


def _make_unique_run_dir(output: str, query_json: str) -> Path:
    """Create a new per-run output directory next to the requested output file.

    The command-line interface stays unchanged: users can keep passing the same
    `--output` and `--query_json` paths. On every run we create a fresh folder
    under the common parent directory and save both files inside it using their
    original file names.
    """

    output_path = Path(output)
    query_json_path = Path(query_json)
    base_dir = output_path.parent if output_path.parent == query_json_path.parent else output_path.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_dir / f"paper_fig_6rows_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _world_xy_to_pixel(xy: tuple[float, float], center_xy: tuple[float, float], bev_size: float, img_size: int = 1024) -> tuple[int, int]:
    """Convert world XY coordinates to BEV image pixel coordinates."""

    cx, cy = center_xy
    half = bev_size / 2.0
    scale = img_size / bev_size
    px = (xy[0] - (cx - half)) * scale
    py = img_size - (xy[1] - (cy - half)) * scale
    return int(round(px)), int(round(py))


def render_record_row(record, all_cells: dict[str, object], bev_size: float = 50.0, topk: int = 5) -> np.ndarray:
    """Render one horizontal row for a single retrieval record."""

    # Left GT panel: force rendering from the original dataset cell instead of
    # `record.query_points_xy`, because that field can be empty in retrieval JSONs.
    gt_cell = _find_query_cell(record, all_cells)
    if gt_cell is not None:
        gt_panel = _render_dataset_cell_panel(gt_cell, record.gt_xy, bev_size)
    else:
        gt_panel = np.ones((1024, 1024, 3), dtype=np.uint8) * 255

    cv2.circle(gt_panel, (512, 512), 18, (0, 0, 255), thickness=-1)
    gt_panel = _crop_and_resize_panel(gt_panel)
    _add_distance_box(gt_panel, 0.0, True, label="GT")

    panels = [gt_panel]

    for cand in record.candidates[:topk]:
        cand_cell = all_cells[str(cand.cell_id)]
        cpts, ccols = _dataset_object_points(cand_cell)
        center = cand.center_xy
        panel = render_bev_from_points(
            cpts,
            ccols,
            center,
            size_m=bev_size,
            img_size=1024,
            point_radius=7,
            query_marker=False,
        )

        # Red: true GT position in this candidate-centered panel.
        gt_px = _world_xy_to_pixel(record.gt_xy, center, bev_size)
        cv2.circle(panel, gt_px, 18, (0, 0, 255), thickness=-1)

        # Blue: fine prediction if available, otherwise coarse cell center.
        pred_xy = cand.fine_xy if cand.fine_xy is not None else cand.center_xy
        pred_px = _world_xy_to_pixel(pred_xy, center, bev_size)
        cv2.circle(panel, pred_px, 18, (255, 0, 0), thickness=-1)

        pred_distance = float(np.linalg.norm(np.asarray(pred_xy) - np.asarray(record.gt_xy)))
        panel = _crop_and_resize_panel(panel)
        _add_distance_box(panel, pred_distance, pred_distance <= 10.0, label="D")
        panels.append(panel)

    gap = 70
    height = panels[0].shape[0]
    width = sum(im.shape[1] for im in panels) + gap * (len(panels) - 1)
    row = np.ones((height, width, 3), dtype=np.uint8) * 255

    x = 0
    for panel_idx, im in enumerate(panels):
        h, w = im.shape[:2]
        row[0:h, x:x + w] = im
        if panel_idx == 0:
            dist_gt_to_center = 0.0
        else:
            dist_gt_to_center = float(np.linalg.norm(np.asarray(record.gt_xy) - np.asarray(record.candidates[panel_idx - 1].center_xy)))
        box_color = (0, 255, 0) if dist_gt_to_center <= 10.0 else (0, 0, 255)
        cv2.rectangle(row, (x, 0), (x + w - 1, h - 1), box_color, thickness=30)
        x += w + gap

    return row


def render_six_query_grid(
    base_path: str,
    scene: str,
    retrieval_json: str,
    output: str,
    query_json: str,
    query_indices: Sequence[int],
    bev_size: float = 50.0,
    topk: int = 5,
    row_gap: int = 90,
) -> None:
    """Render multiple retrieval rows and save their raw query texts as JSON."""

    run_dir = _make_unique_run_dir(output, query_json)
    output_path = run_dir / Path(output).name
    query_json_path = run_dir / Path(query_json).name

    ds = _load_multiscene_dataset(base_path, scene)
    all_cells = {str(c.id): c for c in ds.all_cells}
    records = load_records(retrieval_json)

    selected_records = [records[idx] for idx in query_indices]
    rows = [render_record_row(record, all_cells, bev_size=bev_size, topk=topk) for record in selected_records]

    width = max(row.shape[1] for row in rows)
    height = sum(row.shape[0] for row in rows) + row_gap * (len(rows) - 1)
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    y = 0
    for row in rows:
        h, w = row.shape[:2]
        canvas[y:y + h, 0:w] = row
        y += h + row_gap

    cv2.imwrite(str(output_path), canvas)

    query_payload = {
        "output_image": str(output_path),
        "query_json": str(query_json_path),
        "retrieval_json": str(retrieval_json),
        "query_indices": list(query_indices),
        "queries": [
            {
                "row": row_idx,
                "query_idx": query_idx,
                "query_id": str(record.query_id),
                "gt_xy": [float(record.gt_xy[0]), float(record.gt_xy[1])],
                "query_text": str(record.query_text),
            }
            for row_idx, (query_idx, record) in enumerate(zip(query_indices, selected_records))
        ],
    }
    with query_json_path.open("w", encoding="utf-8") as f:
        json.dump(query_payload, f, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 6-row paper-style retrieval figure and export query texts.")
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--retrieval_json", required=True)
    parser.add_argument("--output", required=True, help="Output multi-row PNG path.")
    parser.add_argument("--query_json", required=True, help="Output JSON path for the six original text queries.")
    parser.add_argument("--query_indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5], help="Query indices to render. Defaults to the first six records.")
    parser.add_argument("--bev_size", type=float, default=50.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--row_gap", type=int, default=90)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if len(args.query_indices) != 6:
        raise ValueError(f"This paper-grid script expects exactly 6 query indices, got {len(args.query_indices)}.")
    render_six_query_grid(
        args.base_path,
        args.scene,
        args.retrieval_json,
        args.output,
        args.query_json,
        args.query_indices,
        bev_size=args.bev_size,
        topk=args.topk,
        row_gap=args.row_gap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
