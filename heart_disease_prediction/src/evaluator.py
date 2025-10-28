
from data_cleaning import load_data, get_processed_data
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, f1_score, recall_score, precision_score, ConfusionMatrixDisplay

path_to_dataset = r'C:\\Users\\patil\\Documents\\GitHub\\ml_projects\\heart_disease_prediction\\data\\raw\datasets\redwankarimsony\heart-disease-data\versions\6'
filename = 'heart_disease_uci.csv'
path_to_model = r'C:\\Users\\patil\\Documents\\GitHub\\ml_projects\\heart_disease_prediction\\models'
MODEL_FILE = 'rf_classifier.joblib'
path_to_matrix = r'C:\\Users\\patil\\Documents\\GitHub\\ml_projects\\heart_disease_prediction\\results'
THRESHOLD = 0.579

def load_model(filename):
    """ Loads the trained pipeline from saved file."""
    try:
        model = joblib.load(filename)
        return model
    except:
        print('Error in loading model.')
        
def calculate_metrics(y_true, y_pred):
    """ calculates accuracy, sensitivity, specificity, precision, F1-score and ROC AUC"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    specificity = tn / (tn + fp)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
#     curve = pd.DataFrame(data = {'fpr': fpr,'tpr':tpr, 'threshold':thresholds})
#     curve['dist_to_optimal'] = np.sqrt((curve['fpr'] - 0)**2 + (curve['tpr'] - 1)**2)
#     ideal_threshold = curve.iloc[curve['dist_to_optimal'].idxmin()]
#     print(ideal_threshold)

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC AUC': roc_auc,
        'Sensitivity (Recall)': recall_score(y_true, y_pred),
        'Specificity': specificity,
        'Precision': precision_score(y_true, y_pred)
    }
    return metrics
        
def plot_confusion_matrix(y_true, y_pred):
    """ Display confucion matrix plot."""
    disp = ConfusionMatrixDisplay.from_predictions(
    y_true, 
    y_pred, 
    cmap=plt.cm.Blues,
    display_labels=['No HD', 'Yes HD'],
    normalize=None # Set to 'true', 'pred', or 'all' to normalize the counts
    )
    disp.ax_.set_title('Confusion Matrix for Random Forest')
    plt.title(f'Confusion Matrix (Threshold={THRESHOLD})')
    plt.savefig(os.path.join(path_to_matrix, 'confusion_matrix.png'))
    plt.show()
    print("Confusion matrix saved as confusion_matrix.png")
    
if __name__ == '__main__':
    full_file_path = os.path.join(path_to_dataset, filename)
    df = load_data(full_file_path)
    
    _, X_test, _, y_test, _ = get_processed_data(df)
    
    full_file_path = os.path.join(path_to_model, MODEL_FILE)
    model_pipeline = load_model(full_file_path)
    
    if model_pipeline is not None:
        
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]
        
        y_pred_thresholded = (y_proba >= THRESHOLD).astype(int)
        
        metrics = calculate_metrics(y_test, y_pred_thresholded)
        
        print("\n--- Model Evaluation ---")
        for name, value in metrics.items():
            print(f"{name:<25}: {value:.4f}")
            
        plot_confusion_matrix(y_test, y_pred_thresholded)
