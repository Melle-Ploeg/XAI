import pandas as pd

def load_concrete(return_X_y=False):
    df = pd.read_csv('../data/concrete.csv')
    features = df.drop('strength', axis=1)
    if return_X_y==True:
        return features, df['strength']
    else:
        return {'feature_names': list(features.columns)}