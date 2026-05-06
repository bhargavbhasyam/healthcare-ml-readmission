from sklearn.ensemble import GradientBoostingClassifier
from src.config import N_ESTIMATORS, RANDOM_STATE

def get_model():
    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE
    )