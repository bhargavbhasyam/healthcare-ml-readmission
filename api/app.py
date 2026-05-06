from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import json

app = FastAPI()

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Load columns
with open("models/columns.json", "r") as f:
    feature_columns = json.load(f)


# Define input schema
class PatientData(BaseModel):
    age: str | None = None
    gender: str | None = None
    time_in_hospital: int | None = None
    num_lab_procedures: int | None = None
    num_medications: int | None = None


@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.dict()])

    # Fill missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = None

    df = df[feature_columns]

    prob = model.predict_proba(df)[:, 1][0]
    pred = int(prob > 0.20)

    return {
        "prediction": pred,
        "probability": float(prob)
    }