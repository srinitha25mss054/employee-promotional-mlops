from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd


class EmployeeData(BaseModel):
    experience_years: float
    performance_score: float
    training_hours: float


app = FastAPI()
model = joblib.load("models/employee_model.pkl")


@app.get("/")
def home():
    return {"message": "Employee promotion prediction api running"}


@app.post("/predict")
def predict(employee: EmployeeData):
    try:
        input_data = pd.DataFrame([{
            "experience_years": employee.experience_years,
            "performance_score": employee.performance_score,
            "training_hours": employee.training_hours
        }])
        prediction = model.predict(input_data)
        result = "promoted" if prediction[0] == 1 else "not promoted"
        return {"predicted_result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))