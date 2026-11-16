# Credit Card Fraud Detection using Graph Neural Networks (GNNs)

## Project Overview

Developed a Graph Neural Network (GNN)-based fraud detection system using the IEEE-CIS Fraud Detection dataset to identify suspicious credit card transactions through transaction relationships.

Unlike traditional machine learning models that analyze transactions independently, this project leverages graph structures to capture relationships between transactions sharing common card, device, email, and address identifiers.

Implemented a GraphSAGE architecture to learn transaction representations through neighborhood aggregation and classify fraudulent transactions.

## Tech Stack

Programming Language: Python
  Deep Learning: PyTorch
  Graph Learning: PyTorch Geometric (PyG)
  GNN Architecture: GraphSAGE
  Data Processing: Pandas, NumPy
  Graph Construction: NetworkX
  Evaluation: Scikit-learn
  Visualization: Matplotlib, Seaborn
  Development Environment: Google Colab

## Dataset

IEEE-CIS Fraud Detection Dataset

The dataset contains transaction records and identity-related information used to identify fraudulent activities.

- `train_transaction.csv` – Transaction attributes and fraud labels.
- `train_identity.csv` – Identity-related attributes associated with transactions.

## Methodology

### 1. Data Preprocessing
- Loaded and explored transaction and identity datasets.
- Merged transaction and identity records using `TransactionID`.
- Handled missing values and processed numerical and categorical features.
- Applied training-only preprocessing to prevent information leakage.

### 2. Graph Construction
- Represented individual transactions as graph nodes.
- Connected transactions through shared card, device, email, and address identifiers.
- Constructed directed temporal edges from earlier to later transactions.
- Generated node feature vectors and graph edge indices using PyTorch Geometric.

### 3. GraphSAGE Model Development
- Implemented a GraphSAGE architecture using PyTorch Geometric.
- Applied neighborhood aggregation to learn transaction representations.
- Added classification layers for binary fraud detection.
- Generated fraud probabilities for individual transactions.

### 4. Model Training
- Trained the model using binary cross-entropy with logits.
- Addressed class imbalance using positive-class weighting.
- Monitored training and validation loss.
- Used early stopping and validation PR-AUC for model selection.

### 5. Model Evaluation
Evaluated model performance using:
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC (Average Precision)
- Confusion Matrix
  
## Results and Deliverables

- Temporal transaction graph representing relationships between transactions.
- GraphSAGE-based binary fraud classification model.
- Fraud probability predictions for test transactions.
- Model evaluation using classification and ranking metrics.
- Confusion matrix and precision-recall curve visualizations.
- Transaction graph visualization.

## Key Highlights

- Applied graph-based deep learning to financial fraud detection.
- Used GraphSAGE for transaction-level neighborhood aggregation.
- Incorporated shared transaction identifiers to capture relational patterns.
- Implemented chronological train, validation, and test splits.
- Addressed class imbalance through weighted loss optimization.
- Applied leakage-aware feature engineering and temporal graph construction.

## Future Improvements

- Explore heterogeneous graph neural networks with separate relationship types.
- Compare GraphSAGE against traditional machine learning models.
- Experiment with neighbor sampling for larger transaction graphs.
- Optimize classification thresholds using validation data.
- Explore real-time fraud detection and graph-based anomaly detection.
