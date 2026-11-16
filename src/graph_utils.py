from collections import defaultdict

import pandas as pd
import torch

from .config import RELATION_COLUMNS


def build_temporal_edges(frame, max_neighbors_per_key: int):
    """
    Create directed edges from prior -> current transaction.
    Each shared identifier links only to a bounded number of recent prior rows.
    """
    recent_by_key = defaultdict(list)
    sources, targets = [], []

    for current_idx, row in frame.iterrows():
        for col in RELATION_COLUMNS:
            if col not in frame.columns:
                continue
            value = row[col]
            import pandas as pd
            if pd.isna(value):
                continue
            # Include column name so equal values across different fields do not collide.
            key = (col, str(value))
            prior = recent_by_key[key][-max_neighbors_per_key:]
            for source_idx in prior:
                sources.append(source_idx)
                targets.append(current_idx)
            recent_by_key[key].append(current_idx)

    if not sources:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([sources, targets], dtype=torch.long)

