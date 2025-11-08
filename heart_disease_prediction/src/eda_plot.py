
from data_cleaning import load_data, get_processed_data
import os
import matplotlib.pyplot as plt
import seaborn as sns

path_to_dataset = r'C:\\Users\\patil\\Documents\\GitHub\\ml_projects\\heart_disease_prediction\\data\\processed'
path_to_result =  r'C:\\Users\\patil\\Documents\\GitHub\\ml_projects\\heart_disease_prediction\\results'
filename = 'processed_data_py.csv'
figname = 'feature_distributions.jpeg'

numeric_cols = [ 'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'num']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

def plot_features(df, filename):
    """ Plots distributions of all features as well as input features plotted against output feature"""
    n_plots_num = len(numeric_cols) 
    n_cols = 3
    n_rows_num = n_plots_num // n_cols + 1
    
    fig_num, axes_num = plt.subplots(n_rows_num, n_cols, figsize = (5*n_cols, 12*n_rows_num))
    axes_num = axes_num.flatten()
    
    for i, col in enumerate(numeric_cols):
        ax = axes_num[i]
        sns.histplot(df[col], kde = True, ax = ax, color = 'skyblue')
        ax.set_title(f'Distribution of {col}', fontsize = 12)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        
    for i in range(n_plots_num, n_cols*n_rows_num):
        fig_num.delaxes(axes_num[i])
        
    plt.savefig(os.path.join(path_to_result, 'eda_num.png'))
    plt.show()
    
    n_plots_cat = len(categorical_cols)
    n_rows_cat = n_plots_cat // n_cols + 1
    
    
    
    fig_cat, axes_cat = plt.subplots(n_rows_cat, n_cols, figsize = (5*n_cols, 12*n_rows_cat))
    axes_cat = axes_cat.flatten()
    
    for i, col in enumerate(categorical_cols):
        ax = axes_cat[i]
        val_counts = df[col].value_counts()
        sns.barplot(x = val_counts.index, y = val_counts.values, ax = ax, color = 'lightpink')
        ax.set_title(f'Distribution of {col}', fontsize = 12)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x')
    
    for i in range(n_plots_cat, n_cols*n_rows_cat):
        fig_cat.delaxes(axes_cat[i])
    plt.savefig(os.path.join(path_to_result, 'eda_cat.png'))
    plt.show()
if __name__ == '__main__':
    full_file_path = os.path.join(path_to_dataset, filename)
    df = load_data(full_file_path)
    _, _, _, _, _ = get_processed_data(df)
    
    full_file_path = os.path.join(path_to_dataset, figname)
    plot_features(df, full_file_path)
