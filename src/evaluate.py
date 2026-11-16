import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)

from .config import CHECKPOINT_FILE, GRAPH_FILE, OUTPUT_DIR, THRESHOLD
from .model import GraphSAGEClassifier
from .utils import ensure_directories, save_json


def main():
    ensure_directories(OUTPUT_DIR)
    if not GRAPH_FILE.exists() or not CHECKPOINT_FILE.exists():
        raise FileNotFoundError("Run graph construction and training before evaluation.")

    bundle = torch.load(GRAPH_FILE, map_location="cpu", weights_only=False)
    data = bundle["data"]
    checkpoint = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)

    model = GraphSAGEClassifier(checkpoint["input_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(model(data.x, data.edge_index)[data.test_mask])
    y_true = data.y[data.test_mask].numpy()
    y_prob = probabilities.numpy()
    y_pred = (y_prob >= THRESHOLD).astype(int)

    metrics = {
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if len(set(y_true.tolist())) > 1 else None
        ),
    }
    save_json(metrics, OUTPUT_DIR / "test_metrics.json")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Fraud"])
    disp.plot(values_format="d")
    plt.title("Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title("Test Set Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=160)
    plt.close()

    test_ids = data.test_mask.nonzero(as_tuple=True)[0].numpy()
    transaction_ids = bundle["transaction_ids"][test_ids]
    pd.DataFrame({
        "TransactionID": transaction_ids,
        "actual_isFraud": y_true,
        "fraud_probability": y_prob,
        "predicted_isFraud": y_pred,
    }).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
