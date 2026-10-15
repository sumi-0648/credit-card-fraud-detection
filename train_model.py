
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

# Default probability threshold; can be tuned on validation data.
threshold = 0.5
predictions = (probabilities >= threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print(
    "Average Precision (PR-AUC):",
    round(average_precision_score(y_test, probabilities), 4),
)

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "threshold": threshold,
    },
    BASE_DIR / "fraud_model.joblib",
)

print("\nModel saved successfully as fraud_model.joblib")
