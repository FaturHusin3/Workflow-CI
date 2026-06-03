import pandas as pd
import mlflow
import mlflow.sklearn
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("titanic_preprocessed.csv")

# Split feature dan target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Enable MLflow autolog
mlflow.sklearn.autolog()

with mlflow.start_run():

    # Model
    model = RandomForestClassifier(
        random_state=42
    )

    # Training
    model.fit(X_train, y_train)

    # Prediction
    pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    # Print hasil
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Simpan metrics ke JSON
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    # Simpan model
    joblib.dump(model, "model.pkl")

    print("metrics.json berhasil dibuat")
    print("model.pkl berhasil dibuat")
