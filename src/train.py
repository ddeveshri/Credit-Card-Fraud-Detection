import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import (
    CHECKPOINT_FILE, EPOCHS, GRAPH_FILE, LEARNING_RATE, OUTPUT_DIR,
    PATIENCE, SEED, WEIGHT_DECAY
)
from .model import GraphSAGEClassifier
from .utils import ensure_directories, save_json, set_seed


def safe_auc_metrics(y_true, probabilities):
    y_np = y_true.detach().cpu().numpy()
    p_np = probabilities.detach().cpu().numpy()
    result = {"pr_auc": float(average_precision_score(y_np, p_np))}
    result["roc_auc"] = (
        float(roc_auc_score(y_np, p_np)) if len(set(y_np.tolist())) > 1 else None
    )
    return result


def main():
    set_seed(SEED)
    ensure_directories(OUTPUT_DIR)
    if not GRAPH_FILE.exists():
        raise FileNotFoundError("Graph not found. Run `python -m src.graph` first.")

    bundle = torch.load(GRAPH_FILE, map_location="cpu", weights_only=False)
    data = bundle["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GraphSAGEClassifier(data.num_node_features).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_labels = data.y[data.train_mask]
    positives = train_labels.sum().float()
    negatives = (train_labels.numel() - positives).float()
    pos_weight = (negatives / positives.clamp(min=1)).clamp(max=100.0)
    history = []
    best_pr_auc = -1.0
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[data.train_mask].float(),
            data.y[data.train_mask].float(),
            pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)[data.val_mask]
            val_probs = torch.sigmoid(val_logits)
            val_loss = F.binary_cross_entropy_with_logits(
                val_logits.float(), data.y[data.val_mask].float()
            ).item()
            metrics = safe_auc_metrics(data.y[data.val_mask], val_probs)

        row = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "validation_loss": float(val_loss),
            **metrics,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={loss.item():.4f} "
            f"| val_loss={val_loss:.4f} | PR-AUC={metrics['pr_auc']:.4f}"
        )

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_channels": data.num_node_features,
                    "best_validation_pr_auc": best_pr_auc,
                },
                CHECKPOINT_FILE,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print("Early stopping.")
                break

    save_json({"history": history}, OUTPUT_DIR / "training_history.json")
    print(f"Best checkpoint saved to {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
