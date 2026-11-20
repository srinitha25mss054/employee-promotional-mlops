import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# STEP 1: Load data
data = pd.read_csv("dataset/employee_promotional.csv")
print(data)
print(data.head())

# STEP 2: Check missing values
print(data.isnull().sum())
print(data.isnull().sum().sum())

# STEP 3: Clean data
data = data.dropna(subset=["promoted"])

numerical_columns = ["experience_years", "performance_score", "training_hours"]
for column in numerical_columns:
    data[column] = data[column].fillna(data[column].median())

print(data.isnull().sum())

# STEP 4: Remove duplicates
print(data.duplicated().sum())
data = data.drop_duplicates()
print(data)

# STEP 5: Save cleaned data
data.to_csv("dataset/employee_clean.csv", index=False)
print("data saved")

# STEP 6: Convert target to numbers
data["promoted"] = data["promoted"].map({"Yes": 1, "No": 0})
print(data)

# STEP 7: Features and target
x = data[["experience_years", "performance_score", "training_hours"]]
y = data["promoted"]

# STEP 8: Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# STEP 9: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
print("model training completed")

# STEP 10: Evaluate
y_pred = model.predict(x_test)
print(y_test.values)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy * 100, "%")
print(classification_report(y_test, y_pred, zero_division=0))

# STEP 11: Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/employee_model.pkl")
print("success")