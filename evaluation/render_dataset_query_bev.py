"""Paper-style BEV renderer for a single dataset query.

This script renders a colored bird's-eye-view image around one query center
using the real KITTI-360Pose dataset objects. The query center is taken from
an actual Pose, and the local neighborhood is rendered in world coordinates.

Typical usage:

    python -m evaluation.render_dataset_query_bev \
        --base_path ./data/KITTI360Pose/k360_30-10_scG_pd10_pc4_spY_all/ \
        --scene 2013_05_28_drive_0000_sync \
        --query_idx 0 \
        --output ./results/query_0000_bev.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from datapreparation.kitti360pose.utils import SCENE_NAMES_TEST, SCENE_NAMES_VAL, CLASS_TO_COLOR
from dataloading.kitti360pose.base import Kitti360BaseDataset
from datapreparation.kitti360pose.drawing import plot_pose_in_best_cell


def _world_points_from_cell(cell) -> tuple[np.ndarray, np.ndarray]:
    """Convert cell-local normalized object coordinates to world-ish coordinates.

    The dataset stores object points in normalized cell coordinates. We map them
    to the cell bounding box in world coordinates so that BEV cropping in meters
    becomes possible.
    """

    points = []
    colors = []
    x0, y0, z0, x1, y1, z1 = cell.bbox_w
    sx = max(x1 - x0, 1e-6)
    sy = max(y1 - y0, 1e-6)
    sz = max(z1 - z0, 1e-6)
    for obj in cell.objects:
        if getattr(obj, "label", "") == "pad":
            continue
        xyz = np.asarray(obj.xyz, dtype=np.float32)
        xyz_world = xyz.copy()
        xyz_world[:, 0] = x0 + xyz[:, 0] * sx
        xyz_world[:, 1] = y0 + xyz[:, 1] * sy
        if xyz_world.shape[1] > 2:
            xyz_world[:, 2] = z0 + xyz[:, 2] * sz
        points.append(xyz_world)
        c = np.array(CLASS_TO_COLOR.get(obj.label, (128, 128, 128)), dtype=np.uint8)
        colors.append(np.tile(c[None, :], (len(xyz_world), 1)))
    if points:
        return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)


def render_query_bev(cell, pose, size_m: float = 10.0, img_size: int = 1024) -> np.ndarray:
    """Render a 10x10m colored BEV around the query pose."""

    center_xy = tuple(np.asarray(pose.pose_w[:2], dtype=np.float32))
    points_world, colors = _world_points_from_cell(cell)

    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    if len(points_world) > 0:
        cx, cy = center_xy
        half = size_m / 2.0
        mask = (
            (points_world[:, 0] >= cx - half)
            & (points_world[:, 0] <= cx + half)
            & (points_world[:, 1] >= cy - half)
            & (points_world[:, 1] <= cy + half)
        )
        points_world = points_world[mask]
        colors = colors[mask]

        scale = img_size / size_m
        px = (points_world[:, 0] - (cx - half)) * scale
        py = img_size - (points_world[:, 1] - (cy - half)) * scale
        for p, c in zip(np.stack([px, py], axis=1), colors):
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < img_size and 0 <= y < img_size:
                cv2.circle(img, (x, y), 1, (int(c[2]), int(c[1]), int(c[0])), thickness=-1)

    cv2.drawMarker(
        img,
        (img_size // 2, img_size // 2),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=40,
        thickness=3,
    )
    cv2.circle(img, (img_size // 2, img_size // 2), 10, (0, 0, 255), thickness=2)
    cv2.putText(img, f"{pose.cell_id}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, f"center=({center_xy[0]:.2f}, {center_xy[1]:.2f})", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


def save_image(img: np.ndarray, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), img)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a dataset-backed 10x10m BEV for one query.")
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--query_idx", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--size_m", type=float, default=50.0)
    parser.add_argument("--img_size", type=int, default=1024)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ds = Kitti360BaseDataset(args.base_path, args.scene)
    pose = ds.poses[args.query_idx]
    cell = ds.cells_dict[pose.cell_id]
    img = render_query_bev(cell, pose, size_m=args.size_m, img_size=args.img_size)
    save_image(img, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
