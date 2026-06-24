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

import matplotlib.pyplot as plt


def correlation(X: np.array, y: np.array, column: int, feature_names: list,
                name: int, regression=True, n_estimators = 100, max_depth=3, plotting=False) -> None:

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
        #print('Length of each explanation: ', len(explanations[10]))
        cols = [i[0] if i else None for i in explanations]
        for sample_residuos in residuos:
            for col, val in zip(cols, sample_residuos):
                if ~np.isnan(val) and val != 0:
                    counter[col, z] += val / X_test.shape[0]
        #print("Did one!")

    df = df.join(pd.DataFrame(counter), how="right")
    columns = ["Original"] + [f'Random #{i + 1}' for i in range(levels)]
    df.columns = columns
    df = df.set_axis(feature_names + ['correlated'], axis=0)

    if plotting:
        figs, axes = plt.subplots(2, 3, sharex=True, sharey=True)
        for y in range(3):
            graph_names = [columns[y], columns[y+3]]

            pair = df.loc[:, graph_names]
            correlated = list(pair.loc["correlated"])
            pair = pair.drop("correlated")

            pair = pair.sort_values(by=columns[y], key=abs, ascending=False)

            for x in range(2):
                graph_name = columns[y+x*3]
                values = pair.loc[:, graph_name]
                feature_names = values.index.tolist()

                #If more than x features, sum abs values of all the values after x. TODO: Pick good number
                feature_count = len(feature_names)

                #print(feature_names)
                values = list(values)
                
                if feature_count > 7:
                    misc_values = sum(values[7:])
                    values = values[:7] + [misc_values]
                    feature_names = feature_names[:7] + [np.str_("Misc features")]
                    #np.append(feature_names[:7], + np.array(["Misc features"]))
                
                print(feature_names)
                print(values) #Dit is er een teveel????
                print(len(feature_names))
                print(len(values))

                location = (x, y)
                top_off = np.zeros(min(8, len(values)))
                top_off[0] = correlated[x]
                axes[location].bar(feature_names, values, width=0.3, label="original")
                axes[location].bar(feature_names, top_off, width=0.3, label="correlated", bottom=values[0])

                axes[location].set_xticks(axes[location].get_xticks(), labels=feature_names, rotation=45, snap=True, rotation_mode="anchor", ha="right")

                #plt.xticks(axes[location].get_xticks(), feature_names, rotation=45, snap=True, rotation_mode="anchor", ha="right")

        plt.show()

    print(df.to_string())
    df.to_csv(f'./chucheria-feature_contribution-4d79070/data/output/correlation_{name}.csv', index_label='col')


def diabetes(plotting=False):
    X, y = load_diabetes(return_X_y=True)
    feature_names = load_diabetes()['feature_names']

    correlation(X, y, 2, feature_names, 'diabetes', plotting=plotting)


def concrete(plotting=False):

    X, y = load_concrete(return_X_y=True)
    feature_names = load_concrete()['feature_names']
    X = np.array(X)
    y = np.array(y)

    correlation(X, y, 7, feature_names, 'concrete', plotting=plotting)

def housing(plotting=False):
    X, y = fetch_california_housing(return_X_y=True)
    feature_names = fetch_california_housing()['feature_names']
    correlation(X, y, 0, feature_names, 'housing', n_estimators=20, plotting=plotting)

def breasts(plotting=False):
    X, y = load_breast_cancer(return_X_y=True)
    print(X.shape)
    feature_names = list(load_breast_cancer()['feature_names'])
    correlation(X, y, 21, feature_names, 'worst texture', False, n_estimators=200, max_depth=20, plotting=plotting)

def wine(plotting=False):
    X, y = load_wine(return_X_y=True)
    feature_names = load_wine()['feature_names']
    correlation(X, y, 1, feature_names, 'wine', regression=False, n_estimators=30, plotting=plotting)

def cov(plotting=False):
    X, y = fetch_covtype(return_X_y=True)
    feature_names = fetch_covtype()['feature_names']
    correlation(X, y, 10, feature_names, 'cov', False, plotting=plotting)

def stroke(plotting=False):
    df = pd.read_csv('./datasets/healthcare-dataset-stroke-data.csv', index_col=0)
    print(df)
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    df = df.fillna(df.mean())
    X = np.array(df.loc[:, df.columns != 'stroke'])
    y = np.array(df.loc[:, 'stroke'])
    feature_names = df.columns
    correlation(X, y, 1, feature_names, 'stroke', regression=False, n_estimators=20, plotting=plotting)

def stars(plotting=False):
    df = pd.read_csv('./datasets/star_classification.csv', index_col=0)
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    X = np.array(df.loc[:, df.columns != 'class'])
    y = np.array(df.loc[:, 'class'])
    feature_names = df.columns
    correlation(X, y, 1, feature_names, 'stars', regression=False, n_estimators=100, plotting=plotting)

def wine_spanish(plotting=False):
    df = pd.read_csv('./datasets/wines_SPA.csv')
    df = df.drop('country', axis=1)
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    y = np.array(df.loc[:, 'price'])
    df = df.fillna(df.mean())
    df = df.drop('price', axis=1)
    X = np.array(df)
    feature_names = list(df.columns)
    correlation(X, y, 2, feature_names, 'wine', regression=True, n_estimators=100, plotting=plotting)

def heart(plotting=False):
    df = pd.read_csv('./datasets/heart.csv')
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    y = np.array(df.loc[:, 'HeartDisease'])
    df = df.fillna(df.mean())
    df = df.drop('HeartDisease', axis=1)
    X = np.array(df)
    feature_names = list(df.columns)
    correlation(X, y, 2, feature_names, 'heart disease', regression=False, n_estimators=10, plotting=plotting)

if __name__ == '__main__':

    # diabetes()
    # concrete()
    # housing()
    breasts(True)
    # wine()
    # cov()
    # stroke()
    # stars()
    # wine_spanish()
    # heart()