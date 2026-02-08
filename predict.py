from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/Student performance model.pkl")
feature_names = model.feature_names_in_

@app.post("/predict")
def predict(student: dict):

    df = pd.DataFrame([student])
    df = pd.get_dummies(df)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }

@app.get("/")
def home():
    return {"status": "API running"}
