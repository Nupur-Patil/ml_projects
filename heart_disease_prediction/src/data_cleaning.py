
import pandas as pd
import numpy as np
import os

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

def encode_categoricals(df):
    """Loads data from a CSV file."""
    
def get_processed_data(df, test_size=0.2, random_state=42):
    """
    Splits data and applies the preprocessing pipeline structure.
    Returns: X_train, X_test, y_train, y_test, preprocessor
    """
    
    
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
    try:
        imputed_df = impute_categoricals(df)
    except:
        print('Imputation error')
