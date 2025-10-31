"""
FastAPI Backend for Calories Burned Prediction
Provides REST API endpoints for model predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import json
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Calories Burned Prediction API",
    description="API for predicting calories burned during exercise",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and scaler
MODEL_PATH = Path("models")
scaler = None
model = None
metrics = None
feature_names = None

def load_models():
    """Load trained models, scaler, and metrics"""
    global scaler, model, metrics, feature_names
    
    try:
        # Load scaler
        scaler = joblib.load(MODEL_PATH / "scaler.pkl")
        
        # Load metrics to determine best model
        with open(MODEL_PATH / "metrics.json", 'r') as f:
            metrics = json.load(f)
        
        # Load the best model
        best_model_name = metrics['best_model']
        model_file = f"{best_model_name}.pkl"
        model = joblib.load(MODEL_PATH / model_file)
        
        feature_names = metrics['feature_names']
        
        print(f"✓ Loaded {best_model_name} model successfully")
        print(f"✓ Model R² Score: {metrics['models'][best_model_name]['r2_score']:.4f}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run train_model.py first to train the models.")

# Load models on startup
load_models()

# Pydantic models for request/response validation
class ExerciseData(BaseModel):
    """Input data model for prediction"""
    gender: str = Field(..., description="Gender: 'male' or 'female'")
    age: int = Field(..., ge=10, le=100, description="Age in years (10-100)")
    height: float = Field(..., ge=100, le=250, description="Height in cm (100-250)")
    weight: float = Field(..., ge=30, le=200, description="Weight in kg (30-200)")
    duration: int = Field(..., ge=1, le=60, description="Exercise duration in minutes (1-60)")
    heart_rate: int = Field(..., ge=60, le=200, description="Heart rate in bpm (60-200)")
    body_temp: float = Field(..., ge=36.0, le=42.0, description="Body temperature in °C (36-42)")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v.lower() not in ['male', 'female']:
            raise ValueError('Gender must be either "male" or "female"')
        return v.lower()
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "male",
                "age": 30,
                "height": 175.0,
                "weight": 75.0,
                "duration": 30,
                "heart_rate": 110,
                "body_temp": 40.0
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction"""
    calories_burned: float
    model_used: str
    input_data: dict
    confidence_score: float

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    try:
        return FileResponse("static/index.html")
    except:
        return HTMLResponse("""
        <html>
            <body>
                <h1>Calories Burned Prediction API</h1>
                <p>API is running! Visit <a href="/docs">/docs</a> for API documentation.</p>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "best_model": metrics['best_model'],
        "feature_names": feature_names,
        "metrics": metrics['models'][metrics['best_model']],
        "timestamp": metrics['timestamp']
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_calories(data: ExerciseData):
    """
    Predict calories burned based on exercise data
    
    - **gender**: Gender of the person (male/female)
    - **age**: Age in years
    - **height**: Height in centimeters
    - **weight**: Weight in kilograms
    - **duration**: Exercise duration in minutes
    - **heart_rate**: Heart rate in beats per minute
    - **body_temp**: Body temperature in Celsius
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Prepare input data in correct order
        # Expected order: Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp
        gender_encoded = 1 if data.gender == 'male' else 0
        
        input_array = np.array([[
            gender_encoded,
            data.age,
            data.height,
            data.weight,
            data.duration,
            data.heart_rate,
            data.body_temp
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Ensure prediction is positive
        prediction = max(0, prediction)
        
        return PredictionResponse(
            calories_burned=round(float(prediction), 2),
            model_used=metrics['best_model'],
            input_data=data.dict(),
            confidence_score=round(metrics['models'][metrics['best_model']]['r2_score'], 4)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/batch_predict")
async def batch_predict(data_list: list[ExerciseData]):
    """Batch prediction for multiple exercise sessions"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    
    for data in data_list:
        try:
            gender_encoded = 1 if data.gender == 'male' else 0
            input_array = np.array([[
                gender_encoded,
                data.age,
                data.height,
                data.weight,
                data.duration,
                data.heart_rate,
                data.body_temp
            ]])
            
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            
            predictions.append({
                "input_data": data.dict(),
                "calories_burned": round(float(max(0, prediction)), 2)
            })
        except Exception as e:
            predictions.append({
                "input_data": data.dict(),
                "error": str(e)
            })
    
    return {
        "total_predictions": len(predictions),
        "predictions": predictions
    }

# Mount static files (for serving HTML, CSS, JS)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
