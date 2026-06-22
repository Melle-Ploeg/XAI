from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from _ft_contribution import GradientBoostingRegressor
from _load_concrete import load_concrete


def test(X: np.array, y: np.array) -> defaultdict:

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.1, 
                                                   random_state=28)

    reg = GradientBoostingRegressor(random_state=0, n_estimators=10, max_depth=2)
    reg.fit(X_train, y_train)
    _, residuos, explanations = reg.decision_path(X_test)

    counter = defaultdict(float)
    cols = [i[0] if i else None for i in explanations]
    for sample_residuos in residuos:
        for col, val in zip(cols, sample_residuos):
            if ~np.isnan(val) and val != 0:
                counter[col] += val / X_test.shape[0]

    return counter


def diabetes():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    columns = X.columns
    levels = [0, 100, 200, 300, 400]

    n_experiments=100
    counter = np.zeros((X.shape[1], len(levels), n_experiments))
    for j in range(n_experiments):
        for i in range(len(levels)):

            X, y = load_diabetes(return_X_y=True)
            np.random.seed(j)
            noise = np.random.normal(scale=np.std(X[:, 2]) * levels[i]/100, 
                                     size=(X.shape[0], ))
            X[:, 2] += noise

            results = test(X, y)
            for col, val in results.items():
                counter[col, i, j] += val

    mean_df = pd.DataFrame(counter.mean(axis=2),
                           columns = [f'Noise %{i}' for i in levels], 
                           index=columns)
                       
    std_df = pd.DataFrame(counter.std(axis=2),
                          columns = [f'Noise %{i}' for i in levels], 
                          index=columns)

    data = pd.concat([mean_df, std_df], axis=1, keys=['mean', 'std'])
    print(data)
    data.stack().to_csv('data/output/noise_diabetes.csv', 
                        index_label=['col', 'name'])


def concrete():

    X, y = load_concrete(return_X_y=True)
    columns = X.columns
    levels = [0,100,200,300,400]

    n_experiments=100
    counter = np.zeros((X.shape[1],len(levels), n_experiments))
    for j in range(n_experiments):
        for i in range(len(levels)):

            X, y = load_concrete(return_X_y=True)
            X,y = np.array(X), np.array(y)
            np.random.seed(j)
            noise = np.random.normal(scale=np.std(X[:,7])*levels[i]/100, size=(X.shape[0],))
            X[:,7] += noise

            results = test(X, y)
            for col, val in results.items():
                counter[col,i,j]+=val

    mean_df = pd.DataFrame(counter.mean(axis=2),
                           columns = [f'Noise %{i}' for i in levels], 
                           index=columns)
                       
    std_df = pd.DataFrame(counter.std(axis=2),
                          columns = [f'Noise %{i}' for i in levels], 
                          index=columns)

    data = pd.concat([mean_df, std_df], axis=1, keys=['mean', 'std'])
    print(data)
    data.stack().to_csv('data/output/noise_concrete.csv', 
                        index_label=['col', 'name'])


if __name__ == '__main__':

    diabetes()
    concrete()