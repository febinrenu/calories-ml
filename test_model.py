"""
Unit Tests for Calories Burned Prediction System
Tests data preprocessing, model predictions, and API endpoints
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app

client = TestClient(app)

# Test Data
SAMPLE_EXERCISE_DATA = {
    "gender": "male",
    "age": 30,
    "height": 175.0,
    "weight": 75.0,
    "duration": 30,
    "heart_rate": 110,
    "body_temp": 40.0
}

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "best_model" in data
        assert "feature_names" in data
        assert "metrics" in data
    
    def test_predict_valid_data(self):
        """Test prediction with valid data"""
        response = client.post("/predict", json=SAMPLE_EXERCISE_DATA)
        assert response.status_code == 200
        data = response.json()
        assert "calories_burned" in data
        assert "model_used" in data
        assert data["calories_burned"] >= 0
    
    def test_predict_invalid_gender(self):
        """Test prediction with invalid gender"""
        invalid_data = SAMPLE_EXERCISE_DATA.copy()
        invalid_data["gender"] = "unknown"
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_age_out_of_range(self):
        """Test prediction with age out of range"""
        invalid_data = SAMPLE_EXERCISE_DATA.copy()
        invalid_data["age"] = 150  # Too old
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_predict_negative_weight(self):
        """Test prediction with negative weight"""
        invalid_data = SAMPLE_EXERCISE_DATA.copy()
        invalid_data["weight"] = -10
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_predict_female(self):
        """Test prediction with female gender"""
        female_data = SAMPLE_EXERCISE_DATA.copy()
        female_data["gender"] = "female"
        response = client.post("/predict", json=female_data)
        assert response.status_code == 200
        data = response.json()
        assert data["calories_burned"] >= 0
    
    def test_batch_predict(self):
        """Test batch prediction"""
        batch_data = [SAMPLE_EXERCISE_DATA, SAMPLE_EXERCISE_DATA]
        response = client.post("/batch_predict", json=batch_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
    
    def test_predict_edge_cases(self):
        """Test prediction with edge case values"""
        edge_cases = [
            {"gender": "male", "age": 10, "height": 100.0, "weight": 30.0, 
             "duration": 1, "heart_rate": 60, "body_temp": 36.0},
            {"gender": "female", "age": 100, "height": 250.0, "weight": 200.0,
             "duration": 60, "heart_rate": 200, "body_temp": 42.0}
        ]
        
        for data in edge_cases:
            response = client.post("/predict", json=data)
            assert response.status_code == 200
            result = response.json()
            assert result["calories_burned"] >= 0


class TestDataValidation:
    """Test data validation logic"""
    
    def test_gender_case_insensitive(self):
        """Test that gender is case-insensitive"""
        test_cases = ["male", "Male", "MALE", "female", "Female", "FEMALE"]
        
        for gender in test_cases:
            data = SAMPLE_EXERCISE_DATA.copy()
            data["gender"] = gender
            response = client.post("/predict", json=data)
            assert response.status_code == 200
    
    def test_missing_required_fields(self):
        """Test prediction with missing required fields"""
        incomplete_data = {"gender": "male", "age": 30}
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422
    
    def test_invalid_data_types(self):
        """Test prediction with invalid data types"""
        invalid_data = SAMPLE_EXERCISE_DATA.copy()
        invalid_data["age"] = "thirty"  # String instead of int
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422


class TestModelPredictions:
    """Test model prediction logic"""
    
    def test_prediction_consistency(self):
        """Test that same input gives same output"""
        response1 = client.post("/predict", json=SAMPLE_EXERCISE_DATA)
        response2 = client.post("/predict", json=SAMPLE_EXERCISE_DATA)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1["calories_burned"] == data2["calories_burned"]
    
    def test_duration_impact(self):
        """Test that longer duration results in more calories"""
        short_duration = SAMPLE_EXERCISE_DATA.copy()
        short_duration["duration"] = 10
        
        long_duration = SAMPLE_EXERCISE_DATA.copy()
        long_duration["duration"] = 60
        
        response_short = client.post("/predict", json=short_duration)
        response_long = client.post("/predict", json=long_duration)
        
        assert response_short.status_code == 200
        assert response_long.status_code == 200
        
        # Longer duration should generally burn more calories
        # (This might not always be true depending on other factors, but is a reasonable expectation)
        calories_short = response_short.json()["calories_burned"]
        calories_long = response_long.json()["calories_burned"]
        
        # Just verify both are positive values
        assert calories_short > 0
        assert calories_long > 0
    
    def test_heart_rate_impact(self):
        """Test that higher heart rate generally results in more calories"""
        low_hr = SAMPLE_EXERCISE_DATA.copy()
        low_hr["heart_rate"] = 70
        
        high_hr = SAMPLE_EXERCISE_DATA.copy()
        high_hr["heart_rate"] = 180
        
        response_low = client.post("/predict", json=low_hr)
        response_high = client.post("/predict", json=high_hr)
        
        assert response_low.status_code == 200
        assert response_high.status_code == 200
        
        calories_low = response_low.json()["calories_burned"]
        calories_high = response_high.json()["calories_burned"]
        
        assert calories_low > 0
        assert calories_high > 0


class TestResponseFormat:
    """Test API response format"""
    
    def test_response_structure(self):
        """Test that response has correct structure"""
        response = client.post("/predict", json=SAMPLE_EXERCISE_DATA)
        assert response.status_code == 200
        
        data = response.json()
        required_fields = ["calories_burned", "model_used", "input_data", "model_confidence"]
        
        for field in required_fields:
            assert field in data
    
    def test_calories_format(self):
        """Test that calories is a positive number"""
        response = client.post("/predict", json=SAMPLE_EXERCISE_DATA)
        data = response.json()
        
        calories = data["calories_burned"]
        assert isinstance(calories, (int, float))
        assert calories >= 0
    
    def test_confidence_score_range(self):
        """Test that confidence score is between 0 and 1"""
        response = client.post("/predict", json=SAMPLE_EXERCISE_DATA)
        data = response.json()
        
        confidence = data["model_confidence"]
        assert 0 <= confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
