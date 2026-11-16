from collections import defaultdict

import torch
from torch_geometric.data import Data

from .config import GRAPH_FILE, IDENTITY_FILE, PROCESSED_DIR, RELATION_COLUMNS
from .data import build_features, chronological_masks, get_labels, load_transactions
from .graph_utils import build_temporal_edges
from .utils import ensure_directories, set_seed


def main():
    from .config import MAX_NEIGHBORS_PER_KEY

    set_seed(42)
    ensure_directories(PROCESSED_DIR)
    frame = load_transactions()
    train_mask, val_mask, test_mask = chronological_masks(frame)
    x = build_features(frame, train_mask)
    y = get_labels(frame)
    edge_index = build_temporal_edges(frame, MAX_NEIGHBORS_PER_KEY)

    graph = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    graph.num_nodes = len(frame)
    torch.save(
        {
            "data": graph,
            "transaction_ids": frame["TransactionID"].to_numpy(),
            "transaction_dt": frame["TransactionDT"].to_numpy(),
        },
        GRAPH_FILE,
    )

    print(f"Saved graph: {GRAPH_FILE}")
    print(f"Nodes: {graph.num_nodes:,} | Directed edges: {edge_index.size(1):,}")
    print(
        f"Split sizes — train: {int(train_mask.sum())}, "
        f"validation: {int(val_mask.sum())}, test: {int(test_mask.sum())}"
    )
    print(f"Fraud prevalence: {float(y.float().mean()):.4%}")


if __name__ == "__main__":
    main()
