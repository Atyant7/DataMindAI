import pandas as pd

def get_dataset_summary(df):
    
    return {
        "Rows" : df.shape[0],
        "Columns" : df.shape[1],
        "Missing Values" : df.isnull().sum().sum(),
        "Duplicate Rows" : df.duplicated().sum()
    }
