from pathlib import Path

DATA_DIR = Path("..") / "data"

RAW_PATH = DATA_DIR / "raw"
TRAINING_FEATURES_PATH = RAW_PATH / "training_set_features.csv"
TRAINING_LABELS_PATH = RAW_PATH / "training_set_labels.csv"
TEST_FEATURES_PATH = RAW_PATH / "test_set_features.csv"
SUBMISSION_FORMAT_PATH = RAW_PATH / "submission_format.csv"

