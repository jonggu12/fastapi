from fastapi import FastAPI
from pydantic import BaseModel
from mangum import Mangum
import joblib
import pandas as pd

app = FastAPI()
handler = Mangum(app)

# 저장된 모델 로드
model = joblib.load("titanic_model.joblib")

class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float

@app.post("/predict-survival")
async def predict_survival(passenger: Passenger):
    passenger_data = pd.DataFrame([passenger.dict()])
    prediction = model.predict(passenger_data)
    return {"survival_prediction": int(prediction[0])}
