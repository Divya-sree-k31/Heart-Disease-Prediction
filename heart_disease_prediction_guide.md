# Building a Heart Disease Prediction Website with Machine Learning

This comprehensive guide will walk you through creating a web application that predicts heart disease risk using machine learning. The project combines frontend web development with data science techniques to create a useful health tool.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Setting Up the Development Environment](#setting-up-the-development-environment)
4. [Building the Machine Learning Model](#building-the-machine-learning-model)
5. [Creating the Web Interface](#creating-the-web-interface)
6. [Integrating the Model with the Website](#integrating-the-model-with-the-website)
7. [Testing and Evaluation](#testing-and-evaluation)
8. [Deployment Options](#deployment-options)
9. [Extensions and Improvements](#extensions-and-improvements)
10. [Ethical Considerations](#ethical-considerations)

## Project Overview

The heart disease prediction application will:
- Allow users to input their health metrics
- Use machine learning to assess their heart disease risk
- Provide visual feedback and analysis of risk factors
- Educate users about heart disease prevention

This project serves both educational and potentially practical purposes, although it should not replace professional medical advice.

## Prerequisites

To build this project, you should have:

- **Programming Knowledge:**
  - Basic Python for machine learning
  - HTML, CSS, and JavaScript for web development
  
- **Tools:**
  - Code editor (VS Code, Sublime Text, etc.)
  - Python 3.x
  - Git (optional, for version control)
  
- **Packages:**
  - Python: NumPy, pandas, scikit-learn, TensorFlow/Keras, Flask (for backend)
  - JavaScript: Chart.js (for visualizations)

## Setting Up the Development Environment

### Step 1: Create Project Structure

```
heart-disease-prediction/
├── data/
│   └── heart.csv                 # Heart disease dataset
├── model/
│   ├── train_model.py            # Script to train the ML model
│   └── model.json                # Saved model (will be created)
├── static/
│   ├── css/
│   │   └── style.css             # CSS for the website
│   ├── js/
│   │   ├── app.js                # Main application logic
│   │   └── model.js              # Model loading and prediction logic
│   └── images/
├── templates/
│   └── index.html                # Main HTML page
└── app.py                        # Flask server (if using backend)
```

### Step 2: Set Up Python Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn flask
```

## Building the Machine Learning Model

### Step 1: Obtain the Dataset

For heart disease prediction, we'll use the UCI Heart Disease dataset, which contains various health metrics and whether patients have heart disease.

```python
# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
df = pd.read_csv(url, names=column_names, na_values='?')

# Handle missing values
df = df.dropna()

# Convert target to binary (0 = no disease, 1 = disease)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Print dataset information
print(f"Dataset shape: {df.shape}")
print(df['target'].value_counts())
```

### Step 2: Explore and Preprocess the Data

```python
# Continue in train_model.py
import matplotlib.pyplot as plt
import seaborn as sns

# Exploratory Data Analysis
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Heart Disease Features')
plt.savefig('../static/images/correlation_matrix.png')

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

### Step 3: Train the Model

```python
# Continue in train_model.py
# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('../static/images/confusion_matrix.png')

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('../static/images/feature_importance.png')

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# For JavaScript implementation, convert model to JSON
# This is a simplified example - you'll need TensorFlowJS for a real implementation
model_data = {
    'features': list(X.columns),
    'weights': [tree.feature_importances_.tolist() for tree in model.estimators_]
}

with open('model.json', 'w') as f:
    json.dump(model_data, f)

print("Model trained and saved successfully!")
```

### Step 4: Convert to TensorFlow.js (Optional)

For a browser-based application, we can convert our model to TensorFlow.js format:

```bash
# Install the TensorFlow.js converter
pip install tensorflowjs

# Convert your model (if using TensorFlow/Keras)
tensorflowjs_converter --input_format keras path/to/model.h5 path/to/output/folder
```

## Creating the Web Interface

### Step 1: Create the HTML Structure

```html
<!-- index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Prediction</title>
    <link rel="stylesheet" href="static/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Heart Disease Risk Prediction</h1>
            <p>Use machine learning to assess your risk of heart disease</p>
        </header>
        
        <div class="content">
            <div class="tabs">
                <div class="tab active" data-tab="prediction">Prediction Tool</div>
                <div class="tab" data-tab="information">Information</div>
                <div class="tab" data-tab="how-it-works">How It Works</div>
            </div>
            
            <!-- Prediction Form Tab Content -->
            <div id="predictionTab" class="tab-content active">
                <!-- Form inputs will go here -->
                <form id="predictionForm">
                    <!-- Input fields will be added here -->
                </form>
                
                <!-- Results section -->
                <div id="results" class="results">
                    <!-- Prediction results will be displayed here -->
                </div>
            </div>
            
            <!-- Information Tab Content -->
            <div id="informationTab" class="tab-content">
                <!-- Educational content about heart disease -->
            </div>
            
            <!-- How It Works Tab Content -->
            <div id="howItWorksTab" class="tab-content">
                <!-- Technical explanation of the model -->
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="static/js/model.js"></script>
    <script src="static/js/app.js"></script>
</body>
</html>
```

### Step 2: Style with CSS

```css
/* style.css */
:root {
    --primary: #ff5e62;
    --secondary: #ff9966;
    --dark: #333;
    --light: #f9f9f9;
    --success: #28a745;
    --warning: #ffc107;
    --danger: #dc3545;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

body {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    min-height: 100vh;
    padding: 20px;
}

.container {
    max-width: 1000px;
    margin: 0 auto;
    background: white;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    overflow: hidden;
}

header {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    padding: 20px;
    text-align: center;
}

h1 {
    font-size: 28px;
    margin-bottom: 10px;
}

.content {
    display: flex;
    flex-direction: column;
    padding: 20px;
}

/* Tab styling */
.tabs {
    display: flex;
    margin-bottom: 20px;
    border-bottom: 1px solid #ddd;
}

.tab {
    padding: 10px 20px;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.3s ease;
}

.tab.active {
    color: var(--primary);
    border-bottom: 3px solid var(--primary);
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

/* Form styling */
.form-container {
    background: var(--light);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}

h2 {
    margin-bottom: 20px;
    color: var(--dark);
}

.form-row {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    margin-bottom: 20px;
}

.form-group {
    flex: 1;
    min-width: 200px;
}

label {
    display: block;
    margin-bottom: 8px;
    font-weight: 500;
    color: var(--dark);
}

input, select {
    width: 100%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 5px;
    font-size: 16px;
    transition: border 0.3s ease;
}

input:focus, select:focus {
    border-color: var(--primary);
    outline: none;
}

/* Button styling */
.btn {
    padding: 12px 24px;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    font-weight: 600;
    transition: all 0.3s ease;
    display: inline-block;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(255, 94, 98, 0.4);
}

/* Results styling */
.results {
    display: none;
    background: white;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
}

.prediction {
    font-size: 24px;
    font-weight: 700;
    margin: 20px 0;
}

.high-risk {
    color: var(--danger);
}

.low-risk {
    color: var(--success);
}

/* Risk meter styling */
.risk-meter {
    height: 20px;
    background: #eee;
    border-radius: 10px;
    margin: 20px 0;
    overflow: hidden;
    position: relative;
}

.risk-level {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
    transition: width 1s ease;
    border-radius: 10px;
}

/* Responsive design */
@media (max-width: 768px) {
    .form-row {
        flex-direction: column;
        gap: 10px;
    }
    
    .form-group {
        min-width: 100%;
    }
}
```

### Step 3: Add JavaScript Functionality

```javascript
// app.js - Main application logic
document.addEventListener('DOMContentLoaded', () => {
    // Tab switching functionality
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.getAttribute('data-tab');
            
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(t => {
                t.classList.remove('active');
            });
            
            // Show the selected tab content and mark tab as active
            document.getElementById(tabName + 'Tab').classList.add('active');
            tab.classList.add('active');
        });
    });
    
    // Form submission handler
    document.getElementById('predictionForm').addEventListener('submit', (e) => {
        e.preventDefault();
        
        // Collect form data
        const formData = {
            age: parseInt(document.getElementById('age').value),
            sex: parseInt(document.getElementById('sex').value),
            cp: parseInt(document.getElementById('cp').value),
            trestbps: parseInt(document.getElementById('trestbps').value),
            chol: parseInt(document.getElementById('chol').value),
            fbs: parseInt(document.getElementById('fbs').value),
            restecg: parseInt(document.getElementById('restecg').value),
            thalach: parseInt(document.getElementById('thalach').value),
            exang: parseInt(document.getElementById('exang').value),
            oldpeak: parseFloat(document.getElementById('oldpeak').value),
            slope: parseInt(document.getElementById('slope').value),
            ca: parseInt(document.getElementById('ca').value),
            thal: parseInt(document.getElementById('thal').value)
        };
        
        // Show loading spinner
        document.getElementById('loadingSpinner').style.display = 'flex';
        document.getElementById('results').style.display = 'none';
        
        // Make prediction
        predictHeartDisease(formData).then(prediction => {
            // Display results
            displayResults(prediction, formData);
            
            // Hide loading spinner
            document.getElementById('loadingSpinner').style.display = 'none';
            document.getElementById('results').style.display = 'block';
            
            // Scroll to results
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        });
    });
    
    // Reset form button
    document.getElementById('resetBtn').addEventListener('click', () => {
        document.getElementById('predictionForm').reset();
        document.getElementById('results').style.display = 'none';
    });
});

// Display prediction results
function displayResults(prediction, formData) {
    // Update prediction text
    const predictionElement = document.getElementById('predictionResult');
    if (prediction.prediction === "High Risk") {
        predictionElement.innerHTML = `<span class="high-risk">High Risk (${prediction.probability}%)</span>`;
        predictionElement.classList.add('high-risk');
        predictionElement.classList.remove('low-risk');
    } else {
        predictionElement.innerHTML = `<span class="low-risk">Low Risk (${prediction.probability}%)</span>`;
        predictionElement.classList.add('low-risk');
        predictionElement.classList.remove('high-risk');
    }
    
    // Update risk meter
    const riskLevel = document.getElementById('riskLevel');
    riskLevel.style.width = `${prediction.probability}%`;
    
    // Generate risk factors analysis
    const factorsList = document.getElementById('factorsList');
    factorsList.innerHTML = '';
    
    // Create an array of risk factors to analyze
    const riskFactors = analyzeRiskFactors(formData);
    
    // Add risk factors to the list
    riskFactors.forEach(factor => {
        const factorElement = document.createElement('div');
        factorElement.className = `factor-item factor-${factor.risk}`;
        factorElement.textContent = factor.message;
        factorsList.appendChild(factorElement);
    });
    
    // Create chart
    createFeatureChart(formData);
}
```

## Integrating the Model with the Website

### Approach 1: Using a JavaScript Model (Client-side)

```javascript
// model.js - Model prediction logic
async function loadModel() {
    // If using TensorFlow.js
    const model = await tf.loadLayersModel('static/model/model.json');
    return model;
}

// Preprocess input data
function preprocessData(data) {
    // Normalize/scale data similar to how we did in training
    // This is a simplified example
    const normalizedData = {};
    
    // Age normalization (assuming age range 25-80)
    normalizedData.age = (data.age - 25) / (80 - 25);
    
    // Similar normalization for other features...
    
    // Convert to tensor
    return tf.tensor([Object.values(normalizedData)]);
}

// Make prediction
async function predictHeartDisease(data) {
    try {
        // For a simplified JavaScript-only approach without TensorFlow.js
        const riskScore = calculateRiskScore(data);
        
        return {
            score: riskScore,
            prediction: riskScore > 0.5 ? "High Risk" : "Low Risk",
            probability: (riskScore * 100).toFixed(1)
        };
        
        // If using TensorFlow.js:
        /*
        const model = await loadModel();
        const processedData =