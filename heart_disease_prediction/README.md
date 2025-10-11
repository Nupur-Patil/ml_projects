# Heart Disease Prediction: A Comparative Analysis of Classification Models

## 1. Project Overview

This project aims to develop, compare, and interpret several machine learning classification models for the prediction of heart disease using a comprehensive dataset of patient medical attributes. The primary goal is to identify the best-performing and most explainable model for this critical diagnostic task.

## 2. Methodology

The project follows a structured data science approach, leveraging scikit-learn pipelines for robust and reproducible workflow.

### 2.1. Key Phases

1.  **Exploratory Data Analysis (EDA):** Initial data assessment, visualization of feature distributions, and analysis of feature correlations with the target variable.
2.  **Data Preprocessing and Feature Engineering:** Handling missing values, encoding categorical variables, scaling numerical features, and using feature selection techniques.
3.  **Model Development & Evaluation:** Training and tuning a suite of classification models.
4.  **Model Interpretability:** Applying advanced techniques (SHAP) to explain model predictions.

### 2.2. Models Implemented

* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier (RF)**
* **eXtreme Gradient Boosting (XGB)**
* **Support Vector Machine (SVM)**

## 3. Technical Implementation Details

### 3.1. Workflow and Pipelines

* A scikit-learn `Pipeline` is utilized for the entire workflow (preprocessing $\rightarrow$ feature selection $\rightarrow$ model training) to prevent data leakage and ensure fair model comparison.
* Extensive **Hyperparameter Tuning** is performed on each model using techniques like Grid Search or Randomized Search with cross-validation.

### 3.2. Evaluation Metrics

Model performance is rigorously evaluated using a comprehensive set of metrics essential for a classification problem:

* **Classification Report:** Precision, Recall, F1-Score, and Support.
* **Confusion Matrix:** Visualization of true and false positives/negatives.
* **Receiver Operating Characteristic (ROC) Curve & Area Under the Curve (ROC-AUC):** Primary metric for model discrimination power.
* **Precision-Recall Curve:** Essential for evaluating performance on potentially imbalanced datasets.

### 3.3. Model Interpretability and Explainability

* **Feature Importance:** Calculated for tree-based models (RF, XGB) to identify the most predictive features.
* **SHapley Additive exPlanations (SHAP):** Used to provide both global (overall feature impact) and local (individual prediction) explanations for the best-performing model(s).

## 4. Repository Structure
'''text
.
├── data/
│   ├── raw/                # Original, immutable dataset
│   └── processed/          # Cleaned, preprocessed data for modeling
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb # Initial exploration and preprocessing steps
│   └── 02_model_training_and_tuning.ipynb # Model comparison, tuning, and evaluation
├── src/
│   ├── pipeline.py         # Defines the data processing and model pipelines
│   ├── models.py           # Contains model definitions and training logic
│   └── utils.py            # Helper functions (e.g., plotting metrics)
├── artifacts/              # Stores trained models and hyperparameter search results
├── requirements.txt        # Project dependencies
└── README.md               # Project overview and documentation (This file)
'''

## 5. Getting Started

### 5.1. Prerequisites

* Python 3.x
* The required packages can be installed using `pip`.

### 5.2. Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Nupur-Patil/ml_projects.git)
    cd Heart-Disease-Prediction
    ```
2.  Create and activate a virtual environment (optional but recommended).
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 5.3. Running the Project

1.  **EDA and Preprocessing:** Run the Jupyter notebooks in the `notebooks/` directory sequentially.
    ```bash
    jupyter notebook
    ```
2.  **Model Training:** Execute the main training script (if created in `src`):
    ```bash
    python src/train_pipeline.py
    ```

## 6. Results and Conclusions

