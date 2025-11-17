"""
FastAPI service for Pet Preference Prediction
Serves predictions using trained XGBoost model
"""

import pickle
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, List
import numpy as np
import os
from pathlib import Path


# Load the model and preprocessing objects
MODEL_FILE = 'pet_preference_model.bin'

try:
    with open(MODEL_FILE, 'rb') as f_in:
        model, label_encoder, dv, target_labels = pickle.load(f_in)
    print(f"Model loaded successfully from {MODEL_FILE}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


def predict_xgboost(model, label_encoder, X, target_labels=['Neither', 'Cat', 'Both', 'Dog']):
    """
    Make predictions using trained XGBoost model.

    Parameters:
    -----------
    model : xgboost.Booster
        Trained XGBoost model
    label_encoder : LabelEncoder
        Fitted label encoder from training
    X : array-like
        Features (already transformed by DictVectorizer)
    target_labels : list
        List of class labels in order

    Returns:
    --------
    predictions : array
        Predicted class labels
    probabilities : array
        Predicted probabilities for each class
    """
    import xgboost as xgb

    # Create DMatrix
    dmatrix = xgb.DMatrix(X)

    # Get probability predictions
    y_pred_proba = model.predict(dmatrix)

    # Get class predictions
    y_pred_encoded = y_pred_proba.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    return y_pred, y_pred_proba


# Define request schema
class PersonFeatures(BaseModel):
    """
    Input features for pet preference prediction

    Features:
    - SEX: Gender (e.g., 'male', 'female')
    - AGE: Age in years (integer)
    - TIP: Type of settlement (e.g., 'city_large', 'city_medium', 'rural')
    - TV: TV watching frequency (e.g., 'daily', 'weekly', 'never')
    - d1: Internet usage frequency (e.g., 'daily', 'weekly', 'never')
    - EDU: Education level (e.g., 'higher', 'secondary', 'incomplete_higher')
    - DOHOD_0: Financial situation (e.g., 'good', 'average', 'bad')
    - PROF1: Main occupation (e.g., 'employee', 'self_employed', 'student', 'retired')
    - PROF2: Organization type (e.g., 'government', 'private', 'unknown')
    - PROF3: Industry (e.g., 'education', 'healthcare', 'trade', 'unknown')
    """
    SEX: str = Field(..., description="Gender")
    AGE: int = Field(..., description="Age in years", ge=0, le=120)
    TIP: str = Field(..., description="Type of settlement")
    TV: str = Field(..., description="TV watching frequency")
    d1: str = Field(..., description="Internet usage frequency")
    EDU: str = Field(..., description="Education level")
    DOHOD_0: str = Field(..., alias="DOHOD-0", description="Financial situation")
    PROF1: str = Field(..., description="Main occupation")
    PROF2: str = Field(..., description="Organization type")
    PROF3: str = Field(..., description="Industry field")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "SEX": "male",
                "AGE": 35,
                "TIP": "city_large",
                "TV": "daily",
                "d1": "daily",
                "EDU": "higher",
                "DOHOD-0": "average",
                "PROF1": "employee",
                "PROF2": "private",
                "PROF3": "it"
            }
        }


# Define response schema
class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted pet preference class")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")


# Initialize FastAPI app
app = FastAPI(
    title="Pet Preference Prediction API",
    description="Predict whether a person is likely to be a cat person, dog person, both, or neither",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the HTML interface"""
    html_file = Path(os.path.dirname(__file__)) / "index.html"
    return html_file.read_text()


@app.post("/predict", response_model=PredictionResponse)
def predict(person: PersonFeatures):
    """
    Predict pet preference for a person based on their demographic and lifestyle features

    Returns the predicted class and probabilities for all classes
    """
    try:
        # Convert input to dictionary format expected by DictVectorizer
        person_dict = person.model_dump(by_alias=True)

        # Transform features using DictVectorizer
        X = dv.transform([person_dict])

        # Make prediction
        y_pred, y_pred_proba = predict_xgboost(model, label_encoder, X, target_labels)

        # Create probability dictionary
        # IMPORTANT: Map probabilities using label_encoder to get correct indices
        probabilities = {}
        for label in target_labels:
            label_idx = label_encoder.transform([label])[0]
            probabilities[label] = float(y_pred_proba[0][label_idx])

        return PredictionResponse(
            prediction=y_pred[0],
            probabilities=probabilities
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch")
def predict_batch(people: List[PersonFeatures]):
    """
    Predict pet preferences for multiple people

    Returns predictions and probabilities for all input samples
    """
    try:
        # Convert all inputs to dictionary format
        people_dicts = [person.model_dump(by_alias=True) for person in people]

        # Transform features using DictVectorizer
        X = dv.transform(people_dicts)

        # Make predictions
        y_pred, y_pred_proba = predict_xgboost(model, label_encoder, X, target_labels)

        # Create response for each person
        results = []
        for i in range(len(people)):
            probabilities = {}
            for label in target_labels:
                label_idx = label_encoder.transform([label])[0]
                probabilities[label] = float(y_pred_proba[i][label_idx])
            
            results.append({
                "prediction": y_pred[i],
                "probabilities": probabilities
            })

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

