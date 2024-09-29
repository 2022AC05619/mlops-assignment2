import numpy as np
import pandas as pd
import feature_eng
import auto_ML

def main_func(file_name):
    df = feature_eng.feature(file_name)
    best_model = auto_ML.Auto_ML(df)

if __name__ == '__main__':
    main_func('../data/covtype.csv')