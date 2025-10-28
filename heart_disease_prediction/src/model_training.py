
from sklearn.pipeline import Pipeline
from data_cleaning import load_data, get_processed_data
from sklearn.ensemble import RandomForestClassifier
import os
import joblib

path_to_dataset = r'C:\\Users\\patil\\Documents\\GitHub\\ml_projects\\heart_disease_prediction\\data\\raw\datasets\redwankarimsony\heart-disease-data\versions\6'
filename = 'heart_disease_uci.csv'
path_to_model = r'C:\\Users\\patil\\Documents\\GitHub\\ml_projects\\heart_disease_prediction\\models'
MODEL_FILE = 'rf_classifier.joblib'

def train_model(X_train, y_train, preprocessor):
    """
    Creates and trains the full pipeline: Preprocessor + Random Forest Classifier.
    
    We use the best parameters found during hyperparameter tuning.
    """
    rf_model = RandomForestClassifier(n_estimators=50,      
        max_depth=6,        
        min_samples_leaf=5,    
        random_state=42,       
        n_jobs=-1)
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_model)])
    print("Fitting the full pipeline (Preprocessor + Random Forest)...")
    full_pipeline.fit(X_train, y_train)
    return full_pipeline

def save_model(model, filename):
    """ Saves the trained model pipeline to a file. """
    joblib.dump(model, filename)
    print(f"\nModel successfully trained and saved as {filename}")

if __name__ == '__main__':
    full_file_path = os.path.join(path_to_dataset, filename)
    df = load_data(full_file_path)
    X_train, _, y_train, _, preprocessor = get_processed_data(df)
    trained_model = train_model(X_train, y_train, preprocessor)
    full_file_path = os.path.join(path_to_model, MODEL_FILE)
    save_model(trained_model, full_file_path)
