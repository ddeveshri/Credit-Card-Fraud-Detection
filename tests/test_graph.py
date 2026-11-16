
import pandas as pd

from src.graph_utils import build_temporal_edges


def test_edges_point_from_earlier_to_later_rows():
    frame = pd.DataFrame({
        "card1": [100, 100, 200],
        "DeviceInfo": ["phone", "phone", "tablet"],
    })
    edges = build_temporal_edges(frame, max_neighbors_per_key=3)
    assert edges.shape[0] == 2
    assert edges.size(1) == 2
    assert (edges[0] < edges[1]).all()


def test_neighbor_limit_is_respected():
    frame = pd.DataFrame({"card1": [7, 7, 7, 7, 7]})
    edges = build_temporal_edges(frame, max_neighbors_per_key=2)
    # Each new node receives edges from at most the two most recent prior rows.
    assert edges.size(1) == 1 + 2 + 2 + 2
