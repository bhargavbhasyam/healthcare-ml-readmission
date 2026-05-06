from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import get_feature_transformer
from src.model_training import get_model
from src.evaluate import evaluate_model
from src.config import DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, MODEL_PATH

import pickle
import json
import os

# 1. Load & clean
df = load_data(DATA_PATH)
df = clean_data(df)

# 2. Split features/target
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

os.makedirs("models", exist_ok=True)

with open("models/columns.json", "w") as f:
    json.dump(X.columns.tolist(), f)

print("Feature columns saved to models/columns.json")

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# 4. Build pipeline
pipeline = Pipeline([
    ("preprocessing", get_feature_transformer(X)),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("model", get_model())
])

# 5. Train
pipeline.fit(X_train, y_train)

# 6. Evaluate
report, auc = evaluate_model(pipeline, X_test, y_test)


# 7. Save metrics (NEW 🔥)
os.makedirs("reports", exist_ok=True)

metrics = {
    "roc_auc": auc,
    "threshold": 0.25
}

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved to reports/metrics.json")

# 8. Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print(f"Model saved to {MODEL_PATH}")