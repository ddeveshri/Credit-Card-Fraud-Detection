from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"
GRAPH_FILE = PROCESSED_DIR / "transaction_graph.pt"
CHECKPOINT_FILE = OUTPUT_DIR / "best_model.pt"

# Set to None to use the full dataset. A cap is useful for an initial run.
MAX_ROWS = 30000

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Relationship columns used to connect transactions. Missing values are ignored.
RELATION_COLUMNS = [
    "card1", "card2", "card3", "card5",
    "addr1", "addr2", "DeviceInfo", "P_emaildomain", "R_emaildomain",
]
MAX_NEIGHBORS_PER_KEY = 5

HIDDEN_CHANNELS = 64
NUM_LAYERS = 2
DROPOUT = 0.30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 20
PATIENCE = 5
SEED = 42
THRESHOLD = 0.50

