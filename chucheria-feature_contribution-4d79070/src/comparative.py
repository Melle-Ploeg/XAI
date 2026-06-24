import numpy as np
from lime import lime_tabular
import pandas as pd
from scipy.sparse import dia
from sklearn.datasets import load_diabetes, fetch_california_housing, load_breast_cancer, load_wine, fetch_covtype
from sklearn.model_selection import train_test_split
import shap

from _ft_contribution import GradientBoostingRegressor
from _ft_contribution_classifier import GradientBoostingClassifier
from _load_concrete import load_concrete

import matplotlib.pyplot as plt

def comparative(X: np.array, y: np.array, feature_names: list, name: str, regression=True, n_estimators=10, plotting = False) -> None:
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.1, 
                                                   random_state=28)

    if regression:
        reg = GradientBoostingRegressor(random_state=0,
                                        n_estimators=n_estimators,
                                        criterion=['friedman_mse', 'squared_error', 'mae'][1],
                                        max_depth=5)
    else:
        reg = GradientBoostingClassifier(random_state=0, max_depth=3, n_estimators=n_estimators)
    reg.fit(X_train, y_train)
    shap_explainer = shap.Explainer(reg)
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train, 
        feature_names=[str(i) for i in range(X.shape[1])],
        verbose=False, mode='regression'
    )

    print("Compare with SHAP and LIME")

    counter_cont = np.zeros(X.shape)
    counter_shap = np.zeros(X.shape)
    counter_lime = np.zeros(X.shape)

    _, residuos, explanations = reg.decision_path(X_test)
    shap_values = shap_explainer(X_test)
    lime_values = []
    for i in range(X_test.shape[0]):
        value = lime_explainer.explain_instance(X_test[i], reg.predict)
        lime_values.append(value.as_map()[0])

    for i in range(X_test.shape[0]):

        residuos_i = residuos[i,:]
        shap_i = shap_values[i,:]
        lime_i = lime_values[i]

        cols = [j[0] if j else None for j in explanations]
        for col, val in zip(cols, residuos_i):
            if ~np.isnan(val) and val != 0:
                counter_cont[i, col] += val

        for col, val in enumerate(shap_i.values):
            counter_shap[i, col] += val

        for col, val in lime_i:
            counter_lime[i, col] += val

    if plotting:
        attributions = {}
        attributions["Contributions"] = []
        attributions["Shapley"] = []
        attributions["LIME"] = []
        
        old_prefix = ""
        cont_avg = shap_avg = lime_avg = 0
        original_feature_names = []

    print("col \t contribution \t shap \t lime")
    for j in range(X.shape[1]):
        feature_name = feature_names[j]

        #Prefix in this context is the original name of the feature. So the one-hot encoded feature collumns 'sex)male' and 'sex)female' simply become 'sex'
        prefix = feature_name.split(")")[0]

        #
        cont_avg += np.mean(np.abs(counter_cont), axis=0)[j]
        shap_avg += np.mean(np.abs(counter_shap), axis=0)[j]
        lime_avg += np.mean(np.abs(counter_lime), axis=0)[j]
        if prefix != old_prefix:
            print(j,"\t",
                cont_avg, "\t",
                shap_avg, "\t",
                lime_avg)
            if plotting:
                attributions["Contributions"] += [cont_avg]
                attributions["Shapley"] += [shap_avg]
                attributions["LIME"] += [lime_avg]

            original_feature_names += [prefix]
            old_prefix = prefix
            cont_avg = shap_avg = lime_avg = 0

    #Construct graph
    if plotting:
        fig, ax = plt.subplots(layout="constrained")
        ax.grouped_bar(attributions, tick_labels=original_feature_names, group_spacing=1)
        ax.set_ylabel('Contribution')
        ax.set_title("Different method's contributions by feature")
        ax.legend(loc='upper left', ncols=3)

        plt.show()

    data = pd.concat([pd.DataFrame(counter_cont),
                      pd.DataFrame(counter_shap),
                      pd.DataFrame(counter_lime)],
                    keys=['contribution', 'shap', 'lime']).reset_index()
    print(data.shape)
    print(original_feature_names)
    data.set_axis(['method', 'obs'] + feature_names, axis=1, copy=False)
    data.to_csv(f'./chucheria-feature_contribution-4d79070/data/output/comparative_{name}.csv', index=False)


def diabetes():
    X, y = load_diabetes(return_X_y=True)
    feature_names = load_diabetes()['feature_names']
    comparative(X, y, feature_names, 'diabetes')


def concrete():
    X, y = load_concrete(return_X_y=True)
    X,y = np.array(X), np.array(y)
    feature_names = load_concrete()['feature_names']
    comparative(X, y, feature_names, 'concrete')

def housing():
    X, y = fetch_california_housing(return_X_y=True)
    feature_names = fetch_california_housing()['feature_names']
    print(feature_names)
    comparative(X, y, feature_names, 'housing', n_estimators=20)

def wine():
    X, y = load_wine(return_X_y=True)
    X,y = np.array(X), np.array(y)
    feature_names = load_wine()['feature_names']
    comparative(X, y, feature_names, 'wine', False)

def stroke():
    df = pd.read_csv('./datasets/healthcare-dataset-stroke-data.csv', index_col=0)
    cat_columns = df.select_dtypes(['object']).columns
    for column in cat_columns:
        dummies = pd.get_dummies(df[column], prefix=column, prefix_sep=")")

        df = pd.concat([df, dummies], axis=1)
        #print(df)
        df = df.drop(labels=column, axis=1)
    #df[cat_columns] = df[cat_columns].apply(lambda x: pd.get_dummies(x)[0])
    #print(df.head())
    df = df.fillna(df.mean())
    y = np.array(df.loc[:, 'stroke'])
    df = df.drop('stroke', axis=1)
    X = np.array(df)
    feature_names = list(df.columns)
    comparative(X, y, feature_names, 'stroke', regression=False, n_estimators=20)

def heart():
    df = pd.read_csv('./datasets/heart.csv')
    cat_columns = df.select_dtypes(['object']).columns
    for column in cat_columns:
        df = pd.concat([df, pd.get_dummies(df[column])], axis=1)
        #print(df)
        df = df.drop(labels=column, axis=1)
    #df[cat_columns] = df[cat_columns].apply(lambda x: pd.get_dummies(x)[0])
    print(df.head())
    y = np.array(df.loc[:, 'HeartDisease'])
    #df = df.fillna(df.mean())
    df = df.drop('HeartDisease', axis=1)
    X = np.array(df)
    feature_names = list(df.columns)
    print(feature_names)
    comparative(X, y, feature_names=feature_names, name='heart disease', regression=False, n_estimators=10)


if __name__ == '__main__': 

    # diabetes()
    concrete()
    # wine()
    # heart()
    stroke()
    # housing()