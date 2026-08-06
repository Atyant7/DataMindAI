import pandas as pd

def load_data(upload_file):
    if upload_file.name.endswith(".csv"):
        return pd.read_csv(upload_file)
    if upload_file.name.endswith(".xlsx"):
        return pd.read_excel(upload_file)