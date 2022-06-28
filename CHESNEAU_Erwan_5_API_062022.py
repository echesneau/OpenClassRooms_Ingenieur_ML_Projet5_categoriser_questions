#!/bin/python3.6

import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

app = FastAPI(debug=False, title="Clusterization of Stackoverflow questions", \
              description="API to predict tags for stackoverflow questions", \
              version="1.0")

var = {}

class FeatureDataInstance(BaseModel):
    """Define JSON data schema for prediction requests."""
    X : str


@app.on_event('startup')
async def load_model():
    var['model'] = joblib.load('pipeline_tfidf_svc.joblib')
    var['encod'] = joblib.load('encoder_1tag.joblib')

@app.get("/")
async def root():
    return {"message": "Welcome to the prediction API"}

@app.post('/predict', status_code=200)
def predict(data: FeatureDataInstance):
    """Generate predictions for data sent to the /predict/ route."""
    prediction = var['model'].predict([data.X]).tolist()
    tag = var['encod'].inverse_transform(prediction)
    return {'tag number': prediction[0],
            'tag' : tag[0] }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', workers=1)
