from pathlib import Path
import tempfile

import numpy as np

from evaluation.render_paper_retrieval_figure import render_query_and_topk
from evaluation.visualize_three_branch_retrieval import RetrievalCandidate, RetrievalRecord, save_records


class _DummyObj:
    def __init__(self, label, xyz, rgb=None):
        self.label = label
        self.xyz = xyz
        self.rgb = rgb


class _DummyCell:
    def __init__(self, cid, shift=0.0):
        self.id = cid
        self.bbox_w = np.array([0.0 + shift, 0.0, 0.0, 10.0 + shift, 10.0, 2.0], dtype=np.float32)
        self.objects = [
            _DummyObj("car", np.array([[0.2, 0.2, 0.1], [0.3, 0.2, 0.1]], dtype=np.float32), np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8)),
            _DummyObj("road", np.array([[0.7, 0.7, 0.1]], dtype=np.float32), np.array([[0, 255, 0]], dtype=np.uint8)),
        ]


class _DummyPose:
    def __init__(self, cell_id):
        self.cell_id = cell_id
        self.pose_w = np.array([5.0, 5.0, 0.0], dtype=np.float32)


class _DummyDS:
    def __init__(self):
        self.poses = [_DummyPose("c0")]
        self.cells = [_DummyCell("c0"), _DummyCell("c1", shift=12.0), _DummyCell("c2", shift=24.0), _DummyCell("c3", shift=36.0), _DummyCell("c4", shift=48.0), _DummyCell("c5", shift=60.0)]
        self.cells_dict = {c.id: c for c in self.cells}


# monkeypatch helper not required; we call a small unit wrapper below

def test_paper_renderer_smoke(monkeypatch=None):
    record = RetrievalRecord(
        query_id="q0",
        query_text="a paper-style query",
        gt_xy=(5.0, 5.0),
        query_points_xy=[],
        candidates=[
            RetrievalCandidate("c1", (17.0, 5.0), 0.9, fine_xy=(17.2, 5.1), fine_score=0.8),
            RetrievalCandidate("c2", (29.0, 5.0), 0.8, fine_xy=(29.1, 5.2), fine_score=0.7),
            RetrievalCandidate("c3", (41.0, 5.0), 0.7, fine_xy=(41.0, 5.0), fine_score=0.6),
            RetrievalCandidate("c4", (53.0, 5.0), 0.6, fine_xy=(53.3, 5.1), fine_score=0.5),
            RetrievalCandidate("c5", (65.0, 5.0), 0.5, fine_xy=(65.2, 5.1), fine_score=0.4),
        ],
    )
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        json_path = d / "retrieval.json"
        save_records([record], json_path)
        out = d / "paper.png"
        # direct call with a temporary monkeypatch via local import swap
        import evaluation.render_paper_retrieval_figure as mod
        old_cls = mod.Kitti360BaseDataset
        try:
            mod.Kitti360BaseDataset = lambda base_path, scene: _DummyDS()
            render_query_and_topk("/tmp", "scene", 0, str(json_path), str(out), bev_size=50.0)
        finally:
            mod.Kitti360BaseDataset = old_cls
        assert out.exists()
