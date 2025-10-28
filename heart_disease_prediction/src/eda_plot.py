
from data_cleaning import load_data, get_processed_data
import os

path_to_dataset = r'C:\\Users\\patil\\Documents\\GitHub\\ml_projects\\heart_disease_prediction\\data\\processed'
filename = 'processed_data_py.csv'
folder_path = r'C:\\Users\\patil\\Documents\\GitHub\\ml_projects\\heart_disease_prediction\\data\\processed'
figname = 'feature_distributions.jpeg'

def plot_features(df, filename):
    """ Plots distributions of all features as well as input features plotted against output feature"""
    n_plots = len(numeric_cols) + len(categorical_cols)
    n_cols = 3
    n_rows = n_plots // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (5*n_cols, 8*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        sns.histplot(df[col], kde = True, ax = ax, color = 'skyblue')
        ax.set_title(f'Distribution of {col}', fontsize = 12)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        
    for i, col in enumerate(categorical_cols):
        ax = axes[i+ len(numeric_cols)]
        val_counts = df[col].value_counts()
        sns.barplot(x = val_counts.index, y = val_counts.values, ax = ax, color = 'lightpink')
        ax.set_title(f'Distribution of {col}', fontsize = 12)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    
    for i in range(n_plots, n_cols*n_rows):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig(file_name)
if __name__ == '__main__':
    full_file_path = os.path.join(path_to_dataset, filename)
    df = load_data(full_file_path)
    _, _, _, _, _ = get_processed_data(df)
    
    print(df['num'].value_counts())
    print(df.isnull().sum())
    full_file_path = os.path.join(folder_path, figname)
    plot_features(df, full_file_path)
