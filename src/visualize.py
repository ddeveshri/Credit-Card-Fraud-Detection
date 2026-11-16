import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch

from .config import GRAPH_FILE, OUTPUT_DIR
from .utils import ensure_directories


def main():
    if not GRAPH_FILE.exists():
        raise FileNotFoundError("Run `python -m src.graph` first.")
    ensure_directories(OUTPUT_DIR)
    bundle = torch.load(GRAPH_FILE, map_location="cpu", weights_only=False)
    data = bundle["data"]

    # Draw a small induced sample so the plot remains readable.
    sample_size = min(250, data.num_nodes)
    edge_index = data.edge_index
    selected = set(range(sample_size))
    graph = nx.DiGraph()
    for node in range(sample_size):
        graph.add_node(node, fraud=int(data.y[node]))
    for src, dst in edge_index.t().tolist():
        if src in selected and dst in selected:
            graph.add_edge(src, dst)

    pos = nx.spring_layout(graph, seed=42, k=0.25)
    labels = [graph.nodes[n]["fraud"] for n in graph.nodes]
    nx.draw_networkx(
        graph, pos, node_size=22, with_labels=False, arrows=False,
        node_color=labels, cmap=plt.cm.coolwarm, width=0.35, alpha=0.8,
    )
    plt.title("Sample of the Temporal Transaction Graph (red/blue = label class)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transaction_graph_sample.png", dpi=180)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'transaction_graph_sample.png'}")


if __name__ == "__main__":
    main()
