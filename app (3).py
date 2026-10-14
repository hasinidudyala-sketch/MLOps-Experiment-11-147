from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("churn_model.pkl")

class Customer(BaseModel):
    feature1: float = Field(ge=-10, le=10)
    feature2: float = Field(ge=-10, le=10)
    feature3: float = Field(ge=-10, le=10)
    feature4: float = Field(ge=-10, le=10)

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Customer Churn API is running"
    }

@app.post("/predict")
def predict(customer: Customer):
    data = [[
        customer.feature1,
        customer.feature2,
        customer.feature3,
        customer.feature4
    ]]

    prediction = int(model.predict(data)[0])

    result = "Likely to churn" if prediction == 1 else "Likely to stay"

    return {
        "status": "success",
        "prediction": result
    }
