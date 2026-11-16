import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from .config import DROPOUT, HIDDEN_CHANNELS, NUM_LAYERS


class GraphSAGEClassifier(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_channels, HIDDEN_CHANNELS))
        for _ in range(max(0, NUM_LAYERS - 1)):
            self.convs.append(SAGEConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS))
        self.classifier = nn.Linear(HIDDEN_CHANNELS, 1)
        self.dropout = DROPOUT

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)
