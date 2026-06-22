import random

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split

from _ft_contribution import GradientBoostingRegressor
from _load_concrete import load_concrete


def correlation(X: np.array, y: np.array, column: int, feature_names: list, 
                name: int) -> None:

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.1, 
                                                   random_state=28)

    reg = GradientBoostingRegressor(random_state=0, n_estimators=10)
    reg.fit(X_train, y_train)
    _, residuos, explanations = reg.decision_path(X_test)

    counter = {i: 0 for i in range(X.shape[1])}
    cols = [i[0] if i else None for i in explanations]
    for sample_residuos in residuos:
        for col, val in zip(cols, sample_residuos):
            if ~np.isnan(val) and val != 0:
                counter[col] += val / X_test.shape[0]

    df = pd.DataFrame(counter.values(), index=counter.keys(), 
                      columns=["Original"])

    print(f"Create new column 10 correlated with column {feature_names[column]}\n")
    alfa = random.random()
    beta = random.random()
    X_train = np.column_stack((X_train, alfa * X_train[:, column] + beta))
    X_test = np.column_stack((X_test, alfa * X_test[:, column] + beta))

    levels = 5
    counter = np.zeros((X_train.shape[1], levels))

    for z in range(levels):
        reg = GradientBoostingRegressor(random_state=z, n_estimators=10)
        reg.fit(X_train, y_train)
        _, residuos, explanations = reg.decision_path(X_test)

        cols = [i[0] if i else None for i in explanations]
        for sample_residuos in residuos:
            for col, val in zip(cols, sample_residuos):
                if ~np.isnan(val) and val != 0:
                    counter[col, z] += val / X_test.shape[0]

    df = df.join(pd.DataFrame(counter), how="right")
    df.columns = ["Original"] + [f'Random #{i + 1}' for i in range(levels)]
    df.set_axis(feature_names + ['correlated'], axis=0, copy=False)
    print(df.to_string())
    df.to_csv(f'../data/output/correlation_{name}.csv', index_label='col')


def diabetes():
    X, y = load_diabetes(return_X_y=True)
    feature_names = load_diabetes()['feature_names']

    correlation(X, y, 2, feature_names, 'diabetes')


def concrete():

    X, y = load_concrete(return_X_y=True)
    feature_names = load_concrete()['feature_names']
    X = np.array(X)
    y = np.array(y)

    correlation(X, y, 7, feature_names, 'concrete')

def housing():
    X, y = fetch_california_housing(return_X_y=True)
    feature_names = fetch_california_housing()['feature_names']
    correlation(X, y, 7, feature_names, 'housing')

if __name__ == '__main__':

    diabetes()
    concrete()
    housing()