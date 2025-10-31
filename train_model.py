"""
Calories Burned Prediction - Model Training Script
This script handles data preprocessing, model training, hyperparameter tuning, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
import json
from datetime import datetime

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def load_and_merge_data():
    """Load and merge exercise and calories datasets"""
    print("Loading datasets...")
    exercise_df = pd.read_csv('dataset/exercise.csv')
    calories_df = pd.read_csv('dataset/calories.csv')
    
    # Merge on User_ID
    df = pd.merge(exercise_df, calories_df, on='User_ID')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def preprocess_data(df):
    """Preprocess the data: handle missing values, encode gender, scale features"""
    print("\nPreprocessing data...")
    
    # Drop User_ID as it's not a feature
    df = df.drop('User_ID', axis=1)
    
    # Check for missing values
    print(f"Missing values before: {df.isnull().sum().sum()}")
    df = df.dropna()
    print(f"Missing values after: {df.isnull().sum().sum()}")
    print(f"Dataset shape after cleaning: {df.shape}")
    
    # Encode gender: male=1, female=0
    df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})
    
    # Separate features and target
    X = df.drop('Calories', axis=1)
    y = df['Calories']
    
    print(f"\nFeatures: {X.columns.tolist()}")
    print(f"Target: Calories")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split data into train/test sets and scale features"""
    print("\nSplitting and scaling data...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n{model_name} Results:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    
    return {
        'model_name': model_name,
        'r2_score': float(r2),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae)
    }

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train baseline Linear Regression model"""
    print("\n" + "="*60)
    print("Training Linear Regression (Baseline)...")
    print("="*60)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    metrics = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    
    # Save model
    joblib.dump(lr_model, 'models/linear_regression.pkl')
    print("Model saved to models/linear_regression.pkl")
    
    return lr_model, metrics

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest with GridSearchCV for hyperparameter tuning"""
    print("\n" + "="*60)
    print("Training Random Forest with GridSearchCV...")
    print("="*60)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Create base model
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        verbose=2,
        n_jobs=-1
    )
    
    print("Performing GridSearchCV... This may take a few minutes.")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    # Get best model
    best_rf_model = grid_search.best_estimator_
    
    metrics = evaluate_model(best_rf_model, X_test, y_test, "Random Forest (Best)")
    metrics['best_params'] = grid_search.best_params_
    metrics['best_cv_score'] = float(grid_search.best_score_)
    
    # Save model
    joblib.dump(best_rf_model, 'models/random_forest.pkl')
    print("Model saved to models/random_forest.pkl")
    
    return best_rf_model, metrics

def save_metrics(lr_metrics, rf_metrics, feature_names):
    """Save all metrics and model info to JSON"""
    metrics_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'feature_names': feature_names.tolist(),
        'models': {
            'linear_regression': lr_metrics,
            'random_forest': rf_metrics
        },
        'best_model': 'random_forest' if rf_metrics['r2_score'] > lr_metrics['r2_score'] else 'linear_regression'
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    print("\nMetrics saved to models/metrics.json")
    print(f"Best model: {metrics_data['best_model']}")

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("CALORIES BURNED PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load and merge data
    df = load_and_merge_data()
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split and scale data
    X_train, X_test, y_train, y_test, feature_names = split_and_scale_data(X, y)
    
    # Train Linear Regression
    lr_model, lr_metrics = train_linear_regression(X_train, X_test, y_train, y_test)
    
    # Train Random Forest with GridSearchCV
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    
    # Save metrics
    save_metrics(lr_metrics, rf_metrics, feature_names)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nModel Comparison:")
    print(f"Linear Regression R²: {lr_metrics['r2_score']:.4f}")
    print(f"Random Forest R²: {rf_metrics['r2_score']:.4f}")
    print(f"\nBest Model: {'Random Forest' if rf_metrics['r2_score'] > lr_metrics['r2_score'] else 'Linear Regression'}")

if __name__ == "__main__":
    main()
