

def analyze_dataset(df):
    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object' , 'category']).columns.tolist()
    analysis = {
        "Numerical Columns" : len(numerical_columns),
        "Categorical Columns" : len(categorical_columns),
        "Missing Values" : df.isnull().sum().sum(),
        "Duplicate Rows" : df.duplicated().sum()
    }
    return analysis