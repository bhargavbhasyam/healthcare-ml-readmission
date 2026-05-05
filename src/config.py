# Data paths
DATA_PATH = "data/raw/diabetic_data.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_data.csv"

# Model path
MODEL_PATH = "models/model.pkl"

# Target column
TARGET_COLUMN = "readmitted"

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model parameters
N_ESTIMATORS = 100
MAX_DEPTH = None

# Drift detection
DRIFT_P_THRESHOLD = 0.05