from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

from _ft_contribution import GradientBoostingRegressor
from _ft_contribution_classifier import GradientBoostingClassifier
from _load_concrete import load_concrete


def something(X: np.array, y: np.array, n_estimators=10, regression=True, max_depth=2) -> defaultdict:

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.1,
                                                   random_state=28)

    if regression:
        reg = GradientBoostingRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth)
    else:
        reg = GradientBoostingClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth)
    reg.fit(X_train, y_train)
    _, residuos, explanations = reg.decision_path(X_test)

    counter = defaultdict(float)
    cols = [i[0] if i else None for i in explanations]
    for sample_residuos in residuos:
        for col, val in zip(cols, sample_residuos):
            if ~np.isnan(val) and val != 0:
                counter[col] += val / X_test.shape[0]

    return counter

def noise(X, y, feature_names, name, column, n_estimators=10, n_experiments=20, regression=True, max_depth=2):
    levels = [0, 100, 200, 300, 400]
    counter = np.zeros((X.shape[1], len(levels), n_experiments))

    for j in range(n_experiments):
        for i in range(len(levels)):

            X_noisy, y_noisy = X.copy(), y.copy()
            np.random.seed(j)
            noise = np.random.normal(scale=np.std(X_noisy[:, column]) * levels[i] / 100,
                                     size=(X_noisy.shape[0],))
            X_noisy[:, column] += noise

            results = something(X_noisy, y_noisy, n_estimators, regression=regression, max_depth=max_depth)
            for col, val in results.items():
                counter[col, i, j] += val

    mean_df = pd.DataFrame(counter.mean(axis=2),
                           columns=[f'Noise %{i}' for i in levels],
                           index=feature_names)

    std_df = pd.DataFrame(counter.std(axis=2),
                          columns=[f'Noise %{i}' for i in levels],
                          index=feature_names)

    data = pd.concat([mean_df, std_df], axis=1, keys=['mean', 'std'])
    print(data)
    data.stack().to_csv(f'../data/output/noise_{name}.csv',
                        index_label=['col', 'name'])

def diabetes():
    X, y = load_diabetes(return_X_y=True)
    feature_names = load_diabetes()['feature_names']
    noise(X, y, feature_names, "diabetes", 2, n_experiments=100)

def concrete():
    X, y = load_concrete(return_X_y=True)
    feature_names = load_concrete()['feature_names']
    X = np.array(X)
    y = np.array(y)
    noise(X, y, feature_names, "concrete", column=7, n_experiments=100)

def housing():
    X, y = fetch_california_housing(return_X_y=True)
    feature_names = fetch_california_housing()['feature_names']
    noise(X, y, feature_names, "housing", column=0, n_estimators=10, n_experiments=10)

def breasts():
    X, y = load_breast_cancer(return_X_y=True)
    feature_names = list(load_breast_cancer()['feature_names'])
    noise(X, y, feature_names, "breasts", column=16, n_estimators=20, max_depth=4, n_experiments=20, regression=False)

def wine():
    X, y = load_wine(return_X_y=True)
    feature_names = load_wine()['feature_names']
    noise(X, y, feature_names, "wine", column=1, n_experiments=20, regression=False)

def stroke():
    df = pd.read_csv('../../datasets/healthcare-dataset-stroke-data.csv', index_col=0)
    y = np.array(df.loc[:, 'stroke'])
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    df = df.fillna(df.mean())
    df = df.drop('stroke', axis=1)
    feature_names = df.columns
    X = np.array(df)
    noise(X, y, feature_names, "stroke", column=1, n_experiments=20, regression=False)

def wine_spanish():
    df = pd.read_csv('../../datasets/wines_SPA.csv')
    df = df.drop('country', axis=1)
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    y = np.array(df.loc[:, 'price'])
    df = df.fillna(df.mean())
    df = df.drop('price', axis=1)
    X = np.array(df)
    feature_names = list(df.columns)
    noise(X, y, feature_names, "wine_spanish", column=2, n_estimators=10, n_experiments=20)

def heart():
    df = pd.read_csv('../../datasets/heart.csv')
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    y = np.array(df.loc[:, 'HeartDisease'])
    df = df.fillna(df.mean())
    df = df.drop('HeartDisease', axis=1)
    X = np.array(df)
    feature_names = list(df.columns)
    noise(X, y, feature_names, "heart", column=2, n_experiments=20, regression=False)

if __name__ == '__main__':

    # diabetes()
    # concrete()
    # housing()
    # wine_spanish()
    # breasts()
    # wine()
    stroke()
    heart()


