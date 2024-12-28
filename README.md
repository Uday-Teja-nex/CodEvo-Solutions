# Machine Learning Internship Tasks (CodEvo Solutions)

This repository contains the solutions to the tasks completed during the **Machine Learning Internship** at CodEvo Solutions. Each task focuses on building hands-on experience in data preprocessing, model development, evaluation, and deployment.

---

## 📋 Tasks Overview

### 1. Wine Quality Prediction
- **Objective**: Predict the quality of wine based on its physicochemical properties.
- **Steps Implemented**:
  - Data cleaning and normalization.
  - Feature engineering and selection.
  - Model training using:
    - Linear Regression
    - Decision Trees
    - Random Forest
    - XGBoost
  - Hyperparameter tuning with GridSearchCV.
- **Key Tools**: Pandas, Scikit-learn, XGBoost, Matplotlib.
- **Output**: Final model saved for deployment, achieving the best accuracy among the tested algorithms.

---

### 2. Customer Churn Prediction
- **Objective**: Identify customers likely to churn based on demographic and usage data.
- **Steps Implemented**:
  - Data preprocessing: Handling missing values and encoding categorical features.
  - Model development:
    - Logistic Regression (baseline).
    - Decision Trees and Random Forest (improved models).
  - Evaluation metrics: Confusion matrix, accuracy, precision, recall.
- **Key Tools**: Pandas, Scikit-learn, Matplotlib, Seaborn.
- **Output**: Identified key features impacting customer churn and saved the best-performing model.

---

### 3. Detecting Spam Emails Using TensorFlow
- **Objective**: Classify emails as spam or ham using machine learning.
- **Steps Implemented**:
  - Data preprocessing: Cleaning, tokenization (using NLTK), and TF-IDF vectorization.
  - Model development:
    - TensorFlow-based deep learning model for text classification.
  - Visualization: Confusion matrix, accuracy plots.
- **Key Tools**: TensorFlow, NLTK, Matplotlib, Scikit-learn.
- **Output**: Successfully classified emails with high accuracy.

---

### 4. Counting Objects Using OpenCV
- **Objective**: Count the number of objects in an image using OpenCV.
- **Steps Implemented**:
  - Image preprocessing: Grayscale conversion, Gaussian blur, and binary thresholding.
  - Contour detection:
    - Used OpenCV to detect and count contours of individual objects.
  - Visualization: Highlighted contours on the original image.
- **Key Tools**: OpenCV, Numpy, Matplotlib.
- **Output**: Counted the number of objects in various images accurately.

---

### 5. Cat & Dog Classification Using CNN
- **Objective**: Build a Convolutional Neural Network (CNN) to classify images as cats or dogs.
- **Steps Implemented**:
  - Data preprocessing:
    - Resized and augmented images with rotation, zoom, and flipping.
  - Model development:
    - Built a CNN using TensorFlow/Keras.
    - Trained the model on augmented training data.
  - Evaluation: Analyzed training/validation accuracy and loss.
- **Key Tools**: TensorFlow, Keras, OpenCV.
- **Output**: Successfully classified cat and dog images with high accuracy.

---

## 🚀 How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Uday-Teja-nex/CodEvo-Solutions.git
