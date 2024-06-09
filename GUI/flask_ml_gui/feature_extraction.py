# feature_extraction.py

import pandas as pd

def get_feature_names():
    df = pd.read_csv('DARWIN.csv')  # Update with the actual path
    df.drop(columns=['ID'], inplace=True)
    feature_names = df.columns.tolist()
    return feature_names
