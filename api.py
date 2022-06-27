#!/bin/python3.6

import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

app = FastAPI(debug=False, title="Predict tags ML", description="API for prediction of tags for stackoverflow questions", version="1.0")

var = {}

class FeatureDataInstance(BaseModel):
    """Define JSON data schema for prediction requests."""
    X: str


@app.on_event('startup')
async def load_model():
    print("startup event")
    var['model'] = joblib.load('pipeline_tfidf_svc.joblib')


@app.get("/")
async def root():
    return {"message": model.predict(["python error value"]).tolist()}

@app.post('/predict', status_code=200)
def predict(data: FeatureDataInstance):
#def predict() :
    """Generate predictions for data sent to the /api/v1/ route."""
    prediction = var['model'].predict([data.X]).tolist()
    return {'y_pred': prediction[0]}

if __name__ == '__main__':
    #model = joblib.load('pipeline_tfidf_svc.joblib')
    #print(model.predict(["python strange value"]))
    uvicorn.run(app, host='127.0.0.1', workers=1)
