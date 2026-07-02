import numpy as np
from lime import lime_tabular
import pandas as pd
from scipy.sparse import dia
from sklearn.datasets import load_diabetes, fetch_california_housing, load_breast_cancer, load_wine, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import shap

from _ft_contribution import GradientBoostingRegressor
from _ft_contribution_classifier import GradientBoostingClassifier
from _load_concrete import load_concrete

import matplotlib.pyplot as plt

def comparative(X: np.array, y: np.array, feature_names: list, name: str, regression=True, n_estimators=10, max_depth=3, plotting = False, single_val = -1, outlier=False) -> None:

    #If single_num is -1, take whole y_test, otherwise pick singular value
    if single_val+1:
        X_test = np.array([X[single_val]])
        y_test = np.array([y[single_val]])
        X = np.concatenate([X[:single_val], X[single_val+1:]])
        y = np.concatenate([y[:single_val], y[single_val+1:]])
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, 
                                                   random_state=28)
    elif outlier:
        X_test = np.array([np.mean(X, axis=0)])
        X_test[0, 1] = np.max(X[:, 1]) + np.std(X[:, 1])
        y_test = np.array([np.max(y) + np.std(y)])#np.array([np.median(y, axis=0)])
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, 
                                                   random_state=28)
        X_train = np.concatenate([X_train, X_test])
        y_train = np.concatenate([y_train, y_test])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, 
                                                   random_state=28)
    


    if regression:
        reg = GradientBoostingRegressor(random_state=0,
                                        n_estimators=n_estimators,
                                        criterion=['friedman_mse', 'squared_error', 'mae'][1],
                                        max_depth=max_depth)
    else:
        reg = GradientBoostingClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth)
    reg.fit(X_train, y_train)
    mask = shap.maskers.Independent(X_train)
    mask  = np.array([np.mean(X_train, axis=0)])
    # print(mask)
    # print(mask.shape)
    # print(X_test.shape)
    # for i in range(20):
    #     print(X_train[0][i].dtype)
    # for i in range(20):
    #     print(X_test[0][i].dtype)
    #print(mask.shape)
    if regression:
        shap_explainer = shap.Explainer(reg)#mask)
    else:
        shap_explainer = shap.Explainer((lambda x: reg.predict_proba(x)[:,1]), mask)
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train, 
        feature_names=[str(i) for i in range(X.shape[1])],
        verbose=False, mode='regression'
    )

    if single_val+1:
        print("Predicted:", reg.predict(X_test)[0])
        print("Predicted odds:", reg.predict_proba(X_test))
        print("Predicted log", reg.predict_log_proba(X_test))

    print("Compare with SHAP and LIME")

    counter_cont = np.zeros(X.shape)
    counter_shap = np.zeros(X.shape)
    counter_lime = np.zeros(X.shape)

    _, residuos, explanations = reg.decision_path(X_test)
    shap_values = shap_explainer(X_test)
    print(shap_values)
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
        attributions["SHAP"] = []
        #attributions["LIME"] = []
        
    old_prefix = ""
    original_feature_names = []
    cont_avg = shap_avg = lime_avg = 0

    print("col \t contribution \t shap \t lime")
    for j in range(X.shape[1]):
        feature_name = feature_names[j]

        #Prefix in this context is the original name of the feature. So the one-hot encoded feature collumns 'sex)male' and 'sex)female' simply become 'sex'
        prefix = feature_name.split(")")[0]

            
        if single_val+1 or outlier:
            cont_avg += counter_cont[0][j]
            shap_avg += counter_shap[0][j]
            lime_avg += counter_lime[0][j]
        else:
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
                attributions["SHAP"] += [shap_avg]
                #attributions["LIME"] += [lime_avg]

            original_feature_names += [prefix]
            old_prefix = prefix
            cont_avg = shap_avg = lime_avg = 0

    #Construct graph
    if plotting:
        fig, ax = plt.subplots(layout="constrained")
        ax.grouped_bar(attributions, tick_labels=original_feature_names, group_spacing=1)
        ax.set_ylabel('Contribution')
        if single_val+1:
            ax.set_title("Different method's contributions at index " + str(single_val) + ", label: " + str(y_test[0]) + ", pred:" + str(np.round(reg.predict_proba(X_test)[0][1], 2)))
        elif outlier:
            ax.set_title("Contributions on outlier")
        else: ax.set_title("Different method's contributions by feature")
        ax.legend(loc='upper left', ncols=3)
        ax.set_xticks(ax.get_xticks(), labels=original_feature_names, rotation=45, snap=True, rotation_mode="anchor", ha="right")

        plt.show()

    data = pd.concat([pd.DataFrame(counter_cont),
                      pd.DataFrame(counter_shap),
                      pd.DataFrame(counter_lime)],
                    keys=['contribution', 'SHAP', 'lime']).reset_index()
    print(data.shape)
    print(original_feature_names)
    data.set_axis(['method', 'obs'] + feature_names, axis=1, copy=False)
    data.to_csv(f'./chucheria-feature_contribution-4d79070/data/output/comparative_{name}.csv', index=False)

# Find a hard-to-classify sample
def find_sample(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ratios = []
    n_neighbours = int(X.shape[0] * 0.99)
    nbrs = NearestNeighbors(n_neighbors=n_neighbours, n_jobs=-1).fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    for i in range(len(X_scaled)):
        nbrs_found = 0
        # Find intra neighbour
        intra = 0
        for j in range(1, len(distances[i])):
            if y[indices[i][j]] == y[i]:
                intra = distances[i][j]
                nbrs_found += 1
                break
        # Find inter neighbour
        inter = 0
        for j in range(len(distances[i])):
            if y[indices[i][j]] != y[i]:
                if distances[i][j] == 0:
                    inter = 100000  # To avoid division by 0
                else:
                    inter = distances[i][j]
                nbrs_found += 1
                break
        if nbrs_found != 2:
            print("WARNING: could not find inter and intra distances for all X, increase n_neighbors")
        ratios.append(intra / inter)
    return np.argmax(ratios)


def diabetes(plotting=False, single_val=-1, outlier=False):
    X, y = load_diabetes(return_X_y=True)
    feature_names = load_diabetes()['feature_names']
    comparative(X, y, feature_names, 'diabetes', max_depth=3, plotting=plotting, single_val=single_val, outlier=outlier)


def concrete(plotting=False, single_val=-1, outlier=False):
    X, y = load_concrete(return_X_y=True)
    X,y = np.array(X), np.array(y)
    feature_names = load_concrete()['feature_names']
    comparative(X, y, feature_names, 'concrete', max_depth=3, plotting=plotting, single_val=single_val, outlier=outlier)
    comparative(X, y, feature_names, 'concrete', max_depth=15, plotting=plotting, single_val=single_val, outlier=outlier)

def housing(plotting=False, single_val=-1):
    X, y = fetch_california_housing(return_X_y=True)
    feature_names = fetch_california_housing()['feature_names']
    print(feature_names)
    comparative(X, y, feature_names, 'housing', n_estimators=20, plotting=plotting, single_val=single_val)

def breasts(plotting=False, single_val=-1):
    X, y = load_breast_cancer(return_X_y=True)
    print(X.shape)
    feature_names = list(load_breast_cancer()['feature_names'])
    comparative(X, y, feature_names, 'breasts', False, n_estimators=200, max_depth=20, plotting=plotting, single_val=single_val)

def wine(plotting=False, single_val=-1):
    X, y = load_wine(return_X_y=True)
    X,y = np.array(X), np.array(y)
    feature_names = load_wine()['feature_names']
    comparative(X, y, feature_names, 'wine', False, plotting=plotting, single_val=single_val)

def stroke(plotting=False, single_val=-1):
    df = pd.read_csv('./datasets/healthcare-dataset-stroke-data.csv', index_col=0)
    cat_columns = df.select_dtypes(['object']).columns
    for column in cat_columns:
        dummies = pd.get_dummies(df[column], prefix=column, prefix_sep=")").map(lambda x: 1 if x else 0)
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

    hard_sample = find_sample(X, y)
    print("Hard to classify sample: ", hard_sample)

    print(y[241])

    comparative(X, y, feature_names, 'stroke', regression=False, n_estimators=20, plotting=plotting)#, single_val=241)

def heart(plotting=False, single_val=-1, outlier=False):
    df = pd.read_csv('./datasets/heart.csv')
    cat_columns = df.select_dtypes(['object']).columns
    for column in cat_columns:
        dummies = pd.get_dummies(df[column], prefix=column, prefix_sep=")").map(lambda x: 1 if x else 0)
        df = pd.concat([df, dummies], axis=1)
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

    hard_sample = find_sample(X, y)
    print("Hard to classify sample: ", hard_sample)

    # print(y[241])

    comparative(X, y, feature_names=feature_names, name='heart disease', regression=False, n_estimators=10, plotting=plotting, single_val=-1, outlier=outlier)


if __name__ == '__main__': 

    # diabetes(True, outlier=True)

    concrete(True, outlier=True)
    # breasts(True)
    # wine(True)
    # heart(True, outlier=True)
    # stroke(True)
    # housing(True)