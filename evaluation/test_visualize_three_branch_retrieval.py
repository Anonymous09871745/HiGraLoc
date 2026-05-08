from pathlib import Path
import tempfile

from evaluation.visualize_three_branch_retrieval import (
    RetrievalCandidate,
    RetrievalRecord,
    build_demo_records,
    crop_points,
    load_records,
    plot_single_query,
    render_records,
    save_records,
)


def test_save_and_load_records_roundtrip():
    records = build_demo_records(num_records=1, topk=3)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "records.json"
        save_records(records, path, metadata={"ok": True})
        loaded = load_records(path)
        assert len(loaded) == 1
        assert loaded[0].query_id == records[0].query_id
        assert loaded[0].candidates[0].cell_id == records[0].candidates[0].cell_id


def test_crop_points_keeps_points_in_range():
    points = [(0.0, 0.0), (10.0, 10.0), (1.0, -1.0), (-6.0, 0.0)]
    out = crop_points(points, (0.0, 0.0), size_m=10.0)
    assert len(out) == 3


def test_plot_single_query_returns_figure():
    record = RetrievalRecord(
        query_id="q1",
        query_text="a short demo query",
        gt_xy=(0.0, 0.0),
        query_points_xy=[(0.1, 0.1), (0.5, -0.2), (1.0, 0.4)],
        candidates=[
            RetrievalCandidate(cell_id="c1", center_xy=(0.0, 0.0), coarse_score=1.0, fine_xy=(0.2, 0.2)),
            RetrievalCandidate(cell_id="c2", center_xy=(2.0, 0.0), coarse_score=0.8, fine_xy=(2.2, 0.2)),
        ],
    )
    fig = plot_single_query(record)
    assert fig is not None
    assert len(fig.axes) >= 4


def test_render_records_creates_png():
    record = RetrievalRecord(
        query_id="q1",
        query_text="a short demo query",
        gt_xy=(0.0, 0.0),
        query_points_xy=[(0.1, 0.1), (0.5, -0.2), (1.0, 0.4)],
        candidates=[
            RetrievalCandidate(cell_id="c1", center_xy=(0.0, 0.0), coarse_score=1.0, fine_xy=(0.2, 0.2)),
            RetrievalCandidate(cell_id="c2", center_xy=(2.0, 0.0), coarse_score=0.8, fine_xy=(2.2, 0.2)),
        ],
    )
    with tempfile.TemporaryDirectory() as d:
        paths = render_records([record], Path(d))
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".png"
