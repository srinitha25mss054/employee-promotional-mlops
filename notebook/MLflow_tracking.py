import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# STEP 1: Load dataset
data = pd.read_csv("dataset/employee_clean.csv")

# STEP 2: Convert target to numbers
data["promoted"] = data["promoted"].map({"Yes": 1, "No": 0})

# STEP 3: Separate features and target
X = data[["experience_years", "performance_score", "training_hours"]]
y = data["promoted"]

# STEP 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 5: Set MLflow experiment
mlflow.set_experiment("Employee_Promotion_Prediction")

# STEP 6: Compare different Random Forest models
tree_values = [50, 100, 200]

for trees in tree_values:

    model = RandomForestClassifier(
        n_estimators=trees,
        random_state=42
    )

    with mlflow.start_run():

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        # Log parameters
        mlflow.log_param("model_name", "Random Forest")
        mlflow.log_param("n_estimators", trees)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(
            model,
            name="employee_promotion_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        # Save the 100-tree model for the API
        if trees == 100:
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/employee_model.pkl")

        print("Trees:", trees)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("--------------------------")

print("All MLflow experiments completed!")