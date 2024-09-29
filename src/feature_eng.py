import pandas as pd
import numpy as np
from autofeat import FeatureSelector
import joblib



def feature(df_file):
    df = pd.read_csv(df_file)

    fsel = FeatureSelector(verbose=2)
    X_new = fsel.fit_transform(df.drop(columns=['Cover_Type']),df['Cover_Type'])

    y_new = df['Cover_Type']

    df_new = X_new
    df_new['Cover_Type'] = df['Cover_Type']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_new, df['Cover_Type'], test_size=0.2, random_state=42)

    df_auto = X_train
    df_auto['Cover_Type'] = df['Cover_Type']

    joblib.dump(fsel, 'trained_feature_selector.joblib')

    return df_auto