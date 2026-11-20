import joblib
import pandas as pd

model = joblib.load("models/employee_model.pkl")

new_employee = pd.DataFrame({
    "experience_years": [7],
    "performance_score": [85],
    "training_hours": [40]
})

prediction = model.predict(new_employee)

if prediction[0] == 1:
    print("employee result: promoted")
else:
    print("employee result: not promoted")