
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

TARGET_COLUMN = 'num'
NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
BINARY_FEATURES = ['sex', 'fbs', 'exang']
OHE_FEATURES = ['cp', 'restecg', 'slope', 'thal']
FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + OHE_FEATURES

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Ensure all necessary columns are present, handling potential missing columns later
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None
    
def impute_group_mode(series):
    """Imputes NA with mode of the categorical series."""
    mode_value = series.mode()
    if mode_value.empty:
        return series.fillna('MISSING_GROUP_MODE')
    return series.fillna(mode_value.iloc[0])

def impute_categoricals(df):
    """Creates imputation requests on relevant columns."""
    df['chol'] = df['chol'].replace(0, np.nan)
    cat_cols_to_impute1 = ['fbs', 'restecg', 'exang']
    cat_cols_to_impute2 = ['slope', 'thal']
    for col in cat_cols_to_impute1:
        df[col] = df.groupby(['age', 'sex'])[col].transform(impute_group_mode)
    for col in cat_cols_to_impute2:
        df[col] = df.groupby(['sex'])[col].transform(impute_group_mode)
    return df

    
def create_preprocessing_pipeline():
    """
    Creates the scikit-learn ColumnTransformer for all data preprocessing steps.
    
    Includes KNNImputer for numerical features, OneHot Encoding for categorical features 
    and StandardScaler.
    """
    
    # 1. Numerical Pipeline: Impute missing values then scale
    numerical_transformer = Pipeline(steps=[
        # Use KNNImputer to fill NaNs, converting to array to avoid feature name warning
        ('imputer', KNNImputer(n_neighbors=5)), 
        # Standardize features (mean=0, variance=1)
        ('scaler', StandardScaler()) 
    ])
    
    # 2. Categorical Pipeline: One-Hot Encode categorical features
    categorical_transformer = Pipeline(steps=[
        # Handle 'unknown' categories and ensure consistent column names
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])
    
    # 3. Column Transformer: Apply pipelines to the correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, OHE_FEATURES)
        ],
        remainder='passthrough',  # Keep any other columns if they exist
        n_jobs=-1
    )
    
    return preprocessor
    
def get_processed_data(df, test_size=0.25, random_state=42):
    """
    Splits data and applies the preprocessing pipeline structure.
    Returns: X_train, X_test, y_train, y_test, preprocessor
    """
    try:
        imputed_df = impute_categoricals(df)
    except:
        print('Imputation error')
    # 1. Handle binary features by replacing values with 1/0
    df['sex'] = np.where(df['sex'] == 'Male', 1, 0)
    df['fbs'] = np.where(df['fbs'] == True, 1, 0)
    df['exang'] = np.where(df['exang'] == True, 1, 0)
    hd = df['num'] > 0
    df.loc[hd, 'num'] = 1
    X = df[FEATURES]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    preprocessor = create_preprocessing_pipeline()
    
    return X_train, X_test, y_train, y_test, preprocessor

def save_data(df,file_path):
    """Saves preprocessed data to a CSV file."""
    try:
        col = FEATURES + [TARGET_COLUMN]
        df = df[col]
        df.to_csv(file_path, index=False)
        # Ensure all necessary columns are present, handling potential missing columns later
        return None
    except FileNotFoundError:
        print(f"Error in saving file at {file_path}")
        return None
if __name__ == '__main__':
    # This block runs if you execute this file directly
    path_to_dataset = r'C:\\Users\\patil\\Documents\\GitHub\\ml_projects\\heart_disease_prediction\\data\\raw\datasets\redwankarimsony\heart-disease-data\versions\6'
    filename = 'heart_disease_uci.csv'
    full_file_path = os.path.join(path_to_dataset, filename)
    
    df = load_data(full_file_path)
    if df is None:
        print(f'Error reading file')
    else:
        print('Data imported')
    
    X_train, X_test, y_train, y_test, preprocessor = get_processed_data(df)
    
    folder_path = r'C:\\Users\\patil\\Documents\\GitHub\\ml_projects\\heart_disease_prediction\\data\\processed'
    filename = 'processed_data_py.csv'
    full_file_path = os.path.join(folder_path, filename)
    save_data(df,full_file_path)
    
    print("\n--- Data Processor Check ---")
    print(f"Training set size: {X_train.shape[0]} samples {X_train.shape[1]} columns")
    print(f"Test set size: {X_test.shape[0]} samples {X_test.shape[1]} columns")
    print("\nPreprocessing Pipeline created successfully.")
