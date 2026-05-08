from pathlib import Path
import tempfile

import cv2
import numpy as np

from evaluation.render_dataset_query_bev import render_query_bev, save_image


class _DummyObj:
    def __init__(self, label, xyz):
        self.label = label
        self.xyz = xyz


class _DummyCell:
    def __init__(self):
        self.id = "dummy_00001"
        self.cell_size = 10.0
        self.bbox_w = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 2.0], dtype=np.float32)
        self.objects = [
            _DummyObj("car", np.array([[0.2, 0.2, 0.1], [0.3, 0.2, 0.1]], dtype=np.float32)),
            _DummyObj("road", np.array([[0.7, 0.7, 0.1]], dtype=np.float32)),
        ]


class _DummyPose:
    def __init__(self):
        self.cell_id = "dummy_00001"
        self.pose_w = np.array([5.0, 5.0, 0.0], dtype=np.float32)


def test_render_query_bev_returns_image():
    img = render_query_bev(_DummyCell(), _DummyPose(), size_m=10.0, img_size=256)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8
    # center marker should make the image non-empty and not pure white
    assert np.any(img != 255)


def test_save_image_writes_file():
    img = render_query_bev(_DummyCell(), _DummyPose(), size_m=10.0, img_size=256)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "bev.png"
        save_image(img, out)
        assert out.exists()
        loaded = cv2.imread(str(out))
        assert loaded is not None
