import requests

url = "http://127.0.0.1:8000/predict"

employee_data = {
    "experience_years": 7,
    "performance_score": 85,
    "training_hours": 40
}

response = requests.post(url, json=employee_data)

print("status code:", response.status_code)
print("prediction:", response.json())