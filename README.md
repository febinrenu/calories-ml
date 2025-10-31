# 🔥 Calories Burned Prediction App

An end-to-end machine learning application that predicts calories burned during exercise based on features like age, gender, exercise duration, heart rate, weight, and body temperature.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 📋 Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Training](#model-training)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Render Deployment](#render-deployment)
- [CI/CD Pipeline](#cicd-pipeline)

## ✨ Features

- **Machine Learning Models**: Linear Regression (baseline) and Random Forest Regressor with hyperparameter tuning
- **RESTful API**: Built with FastAPI for high performance
- **Interactive Web UI**: Clean, responsive interface with sliders and real-time predictions
- **Data Preprocessing**: Automatic scaling, encoding, and missing value handling
- **Model Evaluation**: R², MSE, MAE, and RMSE metrics
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **Docker Support**: Easy containerization and deployment
- **Comprehensive Testing**: Unit tests with pytest

## 🛠️ Tech Stack

**Backend:**
- FastAPI
- Scikit-learn
- Pandas & NumPy
- Joblib
- Pydantic

**Frontend:**
- HTML5
- CSS3
- Vanilla JavaScript

**DevOps:**
- Docker & Docker Compose
- GitHub Actions
- pytest

## 📁 Project Structure

```
calories-ml/
│
├── dataset/
│   ├── calories.csv          # Target variable data
│   └── exercise.csv          # Feature data
│
├── models/
│   ├── linear_regression.pkl # Trained Linear Regression model
│   ├── random_forest.pkl     # Trained Random Forest model
│   ├── scaler.pkl            # Feature scaler
│   └── metrics.json          # Model performance metrics
│
├── static/
│   └── index.html            # Web interface
│
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml   # CI/CD pipeline
│
├── app.py                    # FastAPI application
├── train_model.py            # Model training script
├── test_model.py             # Unit tests
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
└── README.md                 # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip
- Docker (optional)

### Local Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd calories-ml
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the models**
```bash
python train_model.py
```

This will:
- Load and merge the datasets
- Preprocess the data (scaling, encoding)
- Train both Linear Regression and Random Forest models
- Perform hyperparameter tuning with GridSearchCV
- Save trained models and metrics

## 💻 Usage

### Start the Application

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Alternative API Docs: http://localhost:8000/redoc

### Using the Web Interface

1. Open http://localhost:8000 in your browser
2. Select your gender from the dropdown
3. Adjust the sliders for:
   - Age (10-100 years)
   - Height (100-250 cm)
   - Weight (30-200 kg)
   - Duration (1-60 minutes)
   - Heart Rate (60-200 bpm)
   - Body Temperature (36-42°C)
4. Click "Predict Calories Burned"
5. View your predicted calories burned!

## 📚 API Documentation

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

#### 2. Model Information
```http
GET /model/info
```

**Response:**
```json
{
  "best_model": "random_forest",
  "feature_names": ["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"],
  "metrics": {
    "r2_score": 0.9876,
    "mse": 12.34,
    "rmse": 3.51,
    "mae": 2.45
  }
}
```

#### 3. Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "gender": "male",
  "age": 30,
  "height": 175.0,
  "weight": 75.0,
  "duration": 30,
  "heart_rate": 110,
  "body_temp": 40.0
}
```

**Response:**
```json
{
  "calories_burned": 145.32,
  "model_used": "random_forest",
  "input_data": { ... },
  "model_confidence": 0.9876
}
```

#### 4. Batch Prediction
```http
POST /batch_predict
Content-Type: application/json

[
  {
    "gender": "male",
    "age": 30,
    ...
  },
  {
    "gender": "female",
    "age": 25,
    ...
  }
]
```

## 🧠 Model Training

### Features Used:
- **Gender**: Male (1) or Female (0)
- **Age**: Age in years
- **Height**: Height in centimeters
- **Weight**: Weight in kilograms
- **Duration**: Exercise duration in minutes
- **Heart Rate**: Heart rate during exercise (bpm)
- **Body Temperature**: Body temperature during exercise (°C)

### Models:

1. **Linear Regression** (Baseline)
   - Simple linear model
   - Fast training and prediction
   - Good for understanding feature relationships

2. **Random Forest Regressor** (Primary Model)
   - Ensemble learning method
   - GridSearchCV for hyperparameter tuning
   - Parameters tuned:
     - `n_estimators`: [50, 100, 200]
     - `max_depth`: [10, 20, None]
     - `min_samples_split`: [2, 5]
     - `min_samples_leaf`: [1, 2]

### Training Process:

```bash
python train_model.py
```

Output:
```
====================================
CALORIES BURNED PREDICTION - MODEL TRAINING
====================================

Loading datasets...
Dataset shape: (15000, 9)

Preprocessing data...
Missing values after: 0
Dataset shape after cleaning: (15000, 8)

Training Linear Regression...
R² Score: 0.9543

Training Random Forest with GridSearchCV...
Best parameters: {'max_depth': 20, 'min_samples_leaf': 1, ...}
Best CV R² score: 0.9876

Training completed successfully!
Best Model: Random Forest
```

## 🧪 Testing

Run the test suite:

```bash
pytest test_model.py -v
```

**Test Coverage:**
- API endpoint testing
- Data validation
- Model predictions
- Edge cases
- Response format validation

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t calories-predictor .

# Run the container
docker run -p 8000:8000 calories-predictor
```

### Using Docker Compose

```bash
# Start the application
docker-compose up -d

# Stop the application
docker-compose down

# View logs
docker-compose logs -f
```

The application will be available at http://localhost:8000

## 🌐 Render Deployment

### Option 1: Using Web Service

1. **Push your code to GitHub**

2. **Create a new Web Service on Render**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure the service**
   - **Name**: calories-predictor
   - **Environment**: Python 3
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt && python train_model.py
     ```
   - **Start Command**:
     ```bash
     uvicorn app:app --host 0.0.0.0 --port $PORT
     ```

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete
   - Your app will be live at: `https://your-app-name.onrender.com`

### Option 2: Using Docker

1. **Create `render.yaml`** (already configured in the project)

2. **Deploy**
   - Connect repository to Render
   - Render will automatically detect and deploy using Docker

## 🔄 CI/CD Pipeline

The project includes a GitHub Actions workflow that automatically:

1. **On every push/PR:**
   - Installs dependencies
   - Trains the models
   - Runs tests
   - Uploads trained models as artifacts

2. **On push to main:**
   - Builds Docker image
   - Pushes to Docker Hub
   - Ready for deployment

### Setup GitHub Actions:

1. Add these secrets to your GitHub repository:
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Your Docker Hub password

2. Push to main branch to trigger the pipeline

## 📊 Model Performance

Based on the training data, typical performance metrics:

- **R² Score**: ~0.98 (98% variance explained)
- **RMSE**: ~3-5 calories
- **MAE**: ~2-3 calories

The Random Forest model typically outperforms Linear Regression by 3-5% in R² score.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Your Name

## 🙏 Acknowledgments

- Dataset source: Kaggle
- FastAPI documentation
- Scikit-learn documentation
- Render deployment platform

---

**Made with ❤️ and Python**
