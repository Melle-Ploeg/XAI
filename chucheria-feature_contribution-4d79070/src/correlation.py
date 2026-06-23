import random

import numpy as np
import pandas as pd
import sklearn.datasets
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from sklearn.datasets import load_diabetes, fetch_california_housing, load_breast_cancer, load_wine, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from _ft_contribution import GradientBoostingRegressor
from _ft_contribution_classifier import GradientBoostingClassifier
from _ft_contibution_classifier_hist import HistGradientBoostingClassifier
from _load_concrete import load_concrete


def correlation(X: np.array, y: np.array, column: int, feature_names: list,
                name: int, regression=True, n_estimators = 100, max_depth=3) -> None:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                   random_state=28)
    if regression:
        reg = GradientBoostingRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth)
    else:
        reg = GradientBoostingClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    if not regression:
        print("initial accs")
        print(accuracy_score(y_test, y_pred))
    _, residuos, explanations = reg.decision_path(X_test)

    counter = {i: 0 for i in range(X.shape[1])}
    cols = [i[0] if i else None for i in explanations]
    print("Badabie")
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
        if regression:
            reg = GradientBoostingRegressor(random_state=z, n_estimators=n_estimators, max_depth=max_depth)
        else:
            reg = GradientBoostingClassifier(random_state=z, n_estimators=n_estimators, max_depth=max_depth)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print(f"for z = {z}")
        if not regression:
            print(accuracy_score(y_test, y_pred))
        _, residuos, explanations = reg.decision_path(X_test)
        print('Length of each explanation: ', len(explanations[10]))
        cols = [i[0] if i else None for i in explanations]
        for sample_residuos in residuos:
            for col, val in zip(cols, sample_residuos):
                if ~np.isnan(val) and val != 0:
                    counter[col, z] += val / X_test.shape[0]
        print("Did one!")

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
    correlation(X, y, 0, feature_names, 'housing', n_estimators=20)

def breasts():
    X, y = load_breast_cancer(return_X_y=True)
    print(X.shape)
    feature_names = list(load_breast_cancer()['feature_names'])
    correlation(X, y, 16, feature_names, 'breast cancer', False, n_estimators=200, max_depth=20)

def wine():
    X, y = load_wine(return_X_y=True)
    feature_names = load_wine()['feature_names']
    correlation(X, y, 1, feature_names, 'wine', regression=False, n_estimators=30)

def cov():
    X, y = fetch_covtype(return_X_y=True)
    feature_names = fetch_covtype()['feature_names']
    correlation(X, y, 10, feature_names, 'cov', False)

def stroke():
    df = pd.read_csv('../../datasets/healthcare-dataset-stroke-data.csv', index_col=0)
    print(df)
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    df = df.fillna(df.mean())
    X = np.array(df.loc[:, df.columns != 'stroke'])
    y = np.array(df.loc[:, 'stroke'])
    feature_names = df.columns
    correlation(X, y, 1, feature_names, 'stroke', regression=False, n_estimators=20)

def stars():
    df = pd.read_csv('../../datasets/star_classification.csv', index_col=0)
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    X = np.array(df.loc[:, df.columns != 'class'])
    y = np.array(df.loc[:, 'class'])
    feature_names = df.columns
    correlation(X, y, 1, feature_names, 'stars', regression=False, n_estimators=100)

def wine_spanish():
    df = pd.read_csv('../../datasets/wines_SPA.csv')
    df = df.drop('country', axis=1)
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    y = np.array(df.loc[:, 'price'])
    df = df.fillna(df.mean())
    X = np.array(df.drop('price', axis=1))
    feature_names = df.columns
    correlation(X, y, 2, feature_names, 'wine', regression=True, n_estimators=100)

if __name__ == '__main__':

    # diabetes()
    # concrete()
    # housing()
    # breasts()
    wine()
    # cov()
    # stroke()
    # stars()
    # wine_spanish()