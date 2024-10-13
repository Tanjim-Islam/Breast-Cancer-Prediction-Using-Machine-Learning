# Breast Cancer Prediction Using Machine Learning
Specifically:
- Logistic Regression
- SVM
- Decision Trees
- Random Forest
- XGBoost
- Gradient Boosting

## Project Overview
This project focuses on predicting whether a tumor is malignant or benign based on the Breast Cancer Wisconsin dataset. The project includes data preprocessing, exploratory data analysis (EDA), feature selection, and the application of multiple machine learning algorithms such as Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Trees, Random Forest, Gradient Boosting, and XGBoost. The model performance is evaluated using accuracy, confusion matrix, classification report, and ROC curves.

## Requirements
The following libraries are required to run the code:
- Python 3.10
- pandas
- numpy
- seaborn
- matplotlib
- missingno
- scikit-learn
- xgboost

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Tanjim-Islam/Breast-Cancer-Prediction-Using-Machine-Learning.git
    cd breast-cancer-prediction
    ```

2. Install the necessary dependencies:
    ```bash
    pip install pandas numpy seaborn matplotlib missingno scikit-learn xgboost
    ```

3. Ensure the `breast_cancer.csv` dataset is available in the working directory.


## Code Structure

**1. Data Preprocessing:**

**2. Exploratory Data Analysis (EDA):**
   - Plot histograms and density plots for various features.
   - Plot a correlation heatmap to observe multicollinearity between features.

**3. Feature Selection:**
   - Drop features with high correlation (>0.92) to reduce multicollinearity.
   - Select 23 features for training.

**4. Data Splitting and Scaling:**
   - Split the data into training and testing sets.
   - Apply standardization using `StandardScaler` for normalization.

**5. Machine Learning Models:**
   - Implement the following machine learning models:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM) with Grid Search for hyperparameter tuning
     - Decision Tree Classifier with Grid Search for hyperparameter tuning
     - Random Forest Classifier
     - Gradient Boosting Classifier
     - XGBoost Classifier

**6. Model Evaluation:**
   - Evaluate the models using the following metrics:
     - Accuracy score on training and testing data.
     - Confusion matrix.
     - Classification report (precision, recall, F1-score).
     - ROC curves for each model to visualize performance.

**7. Visualization:**
   - Plot ROC curves for all models.
   - Compare model performance based on accuracy and ROC-AUC score.
   - Save performance evaluation plots.

## Results
The results for the models are presented in terms of:
- Accuracy (%)
- ROC AUC (%)

A final bar plot shows the comparison between different machine learning models in terms of their accuracy and ROC-AUC scores.

## Model Deployment
The model that performs the best is saved as `brest_cancer.pkl` using `pickle` for future deployment.

