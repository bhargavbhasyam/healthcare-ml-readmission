from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test, threshold=0.20):
    """
    Evaluate model performance with custom threshold

    Args:
        model: trained pipeline/model
        X_test: test features
        y_test: true labels
        threshold: probability cutoff for classification

    Returns:
        report (str), auc (float)
    """

    # 1. Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # 2. Apply threshold
    preds = (probs > threshold).astype(int)

    # 3. Classification report
    report = classification_report(y_test, preds)

    # 4. ROC-AUC
    auc = roc_auc_score(y_test, probs)

    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Threshold = {threshold})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # 6. Print results (optional but useful)
    print("\nClassification Report:\n", report)
    print("ROC-AUC:", auc)

    return report, auc