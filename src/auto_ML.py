import pandas as pd
import numpy as np
from pycaret.classification import *
import joblib


def Auto_ML(df_auto):
    df_feature_names = df_auto.columns

    clf1 = setup(data = df_auto, target = 'Cover_Type')

    best_model = compare_models()

    # Get feature importance from the model
    feature_importances = best_model.feature_importances_

    # Create a DataFrame for easy visualization
    importance_df = pd.DataFrame({
        'Feature': df_feature_names[:37],
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Display the feature importances
    print(importance_df)
    
    joblib.dump(best_model, 'model.joblib')
    
    return best_model
