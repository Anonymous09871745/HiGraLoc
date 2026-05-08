from pathlib import Path
import tempfile

import numpy as np

from evaluation.render_query_bev import render_bev_from_points, save_bev_image


def test_render_bev_from_points_marks_center():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [4.9, 4.9, 0.0],
        [6.0, 6.0, 0.0],
    ], dtype=np.float32)
    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
    ], dtype=np.uint8)
    img = render_bev_from_points(points, colors, (0.0, 0.0), size_m=10.0, img_size=256)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8
    assert (img[:, :, 2] > 200).any()  # red channel exists for the marker


def test_save_bev_image_writes_file():
    img = np.ones((64, 64, 3), dtype=np.uint8) * 255
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "bev.png"
        save_bev_image(img, out)
        assert out.exists()
