"""Render a colored 10x10m BEV image around one query center.

This is a standalone visualizer for paper-quality inspection. It renders:
- point cloud / object points in a 10x10 meter square around the query center
- a colored BEV image using object class colors or RGB if available
- the query center marked clearly

The script is intentionally independent from the retrieval pipeline so it can
be used for a single query from the dataset or for synthetic unit tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from datapreparation.kitti360pose.utils import CLASS_TO_COLOR


def _world_to_pixel(points_xy: np.ndarray, center_xy: Tuple[float, float], size_m: float, img_size: int) -> np.ndarray:
    half = size_m / 2.0
    cx, cy = center_xy
    scale = img_size / size_m
    px = (points_xy[:, 0] - (cx - half)) * scale
    py = (points_xy[:, 1] - (cy - half)) * scale
    # image y axis should go up in world, down in image
    py = img_size - py
    return np.stack([px, py], axis=1)


def _draw_points(img: np.ndarray, pts_xy: np.ndarray, color: Tuple[int, int, int], radius: int = 1) -> None:
    for p in pts_xy:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), radius, color, thickness=-1)


def render_bev_from_points(
    points_xyz: np.ndarray,
    colors_rgb: Optional[np.ndarray],
    center_xy: Tuple[float, float],
    size_m: float = 10.0,
    img_size: int = 1024,
    query_marker: bool = True,
    point_radius: int = 2,
) -> np.ndarray:
    """Render a colored BEV image from raw points.

    Args:
        points_xyz: Nx3 or Nx2 points in world coordinates.
        colors_rgb: Optional Nx3 colors in RGB or uint8 [0,255]. If None, uses gray.
        center_xy: Center of the BEV crop in world coordinates.
        size_m: Width/height in meters.
        img_size: Output image size in pixels.
    """

    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.size == 0:
        return np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    points_xy = points_xyz[:, :2]
    half = size_m / 2.0
    cx, cy = center_xy
    mask = (
        (points_xy[:, 0] >= cx - half)
        & (points_xy[:, 0] <= cx + half)
        & (points_xy[:, 1] >= cy - half)
        & (points_xy[:, 1] <= cy + half)
    )
    points_xy = points_xy[mask]

    if colors_rgb is None:
        colors_rgb = np.full((len(points_xy), 3), 160, dtype=np.uint8)
    else:
        colors_rgb = np.asarray(colors_rgb)[mask]
        if colors_rgb.max() <= 1.0:
            colors_rgb = (colors_rgb * 255).astype(np.uint8)
        else:
            colors_rgb = colors_rgb.astype(np.uint8)

    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    if len(points_xy) > 0:
        pix = _world_to_pixel(points_xy, center_xy, size_m, img_size)
        for p, c in zip(pix, colors_rgb):
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < img_size and 0 <= y < img_size:
                cv2.circle(img, (x, y), point_radius, (int(c[2]), int(c[1]), int(c[0])), thickness=-1)

    if query_marker:
        qx, qy = _world_to_pixel(np.array([[cx, cy]], dtype=np.float32), center_xy, size_m, img_size)[0]
        # The center marker is intentionally green so it is visually distinct from
        # the red GT marker and the blue prediction marker in the paper figure.
        cv2.drawMarker(img, (int(round(qx)), int(round(qy))), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=60, thickness=4)
        cv2.circle(img, (int(round(qx)), int(round(qy))), 14, (0, 255, 0), thickness=2)

    return img


def render_bev_from_objects(cell, pose_xy: Tuple[float, float], size_m: float = 10.0, img_size: int = 1024) -> np.ndarray:
    """Render a BEV from a Cell object containing Object3d instances."""

    all_points = []
    all_colors = []
    for obj in getattr(cell, "objects", []):
        if getattr(obj, "label", "") == "pad":
            continue
        xyz = np.asarray(obj.xyz, dtype=np.float32)
        rgb = None
        if getattr(obj, "rgb", None) is not None:
            rgb = np.asarray(obj.rgb)
        if rgb is None:
            c = np.array(CLASS_TO_COLOR.get(obj.label, (128, 128, 128)), dtype=np.uint8)
            rgb = np.tile(c[None, :], (len(xyz), 1))
        all_points.append(xyz)
        all_colors.append(rgb)

    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = None

    return render_bev_from_points(points, colors, pose_xy, size_m=size_m, img_size=img_size)


def save_bev_image(img: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 10x10m colored BEV around one query center.")
    parser.add_argument("--output", type=str, required=True, help="Output image path.")
    parser.add_argument("--center_x", type=float, required=True, help="Query center X in world coordinates.")
    parser.add_argument("--center_y", type=float, required=True, help="Query center Y in world coordinates.")
    parser.add_argument("--points_npy", type=str, default=None, help="Optional Nx3 numpy file for raw points.")
    parser.add_argument("--colors_npy", type=str, default=None, help="Optional Nx3 numpy file for RGB colors.")
    parser.add_argument("--size_m", type=float, default=50.0, help="Crop size in meters.")
    parser.add_argument("--img_size", type=int, default=1024, help="Output image size in pixels.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.points_npy is None:
        img = np.ones((args.img_size, args.img_size, 3), dtype=np.uint8) * 255
        cx, cy = args.center_x, args.center_y
        cv2.drawMarker(img, (args.img_size // 2, args.img_size // 2), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=40, thickness=3)
        cv2.putText(img, f"center=({cx:.2f}, {cy:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    else:
        points = np.load(args.points_npy)
        colors = np.load(args.colors_npy) if args.colors_npy else None
        img = render_bev_from_points(points, colors, (args.center_x, args.center_y), size_m=args.size_m, img_size=args.img_size)
    save_bev_image(img, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
