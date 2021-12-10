# file for training and testing features set
""""
L'ensemble d'entraînement est utilisé pour ajuster les modèles ; 
l'ensemble de validation est utilisé pour estimer l'erreur de prédiction pour la sélection de modèle ;
l'ensemble de test est utilisé pour l'évaluation de l'erreur de généralisation du modèle final choisi. 
Idéalement, l'ensemble de test doit être conservé dans un "coffre-fort" et n'être sorti qu'à la fin de l'analyse des données
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold 
from sklearn.svm import LinearSVC

features = pd.read_csv('features.csv')
features_name = features.columns.values.tolist()
X= features.to_numpy()
X= np.nan_to_num(X)
#X_te = pd.read_csv('features_te.csv').to_numpy()
y = pd.read_csv('target.csv').to_numpy()

train_ratio = 0.7
validation_ratio = 0.02
test_ratio = 0.28
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 


def kFold_dataset(X,y, model, nbr_fold = 5):
    score_test = 0
    score_train = 0
    kf = KFold(n_splits = nbr_fold, random_state=None)
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        model.fit(X_train, y_train)
        score_test += model.score(X_test, y_test)
        score_train += 1 - model.score(X_train, y_train)
    
    score_test_mean = score_test/nbr_fold
    score_train_mean = score_train/nbr_fold
    return score_train_mean, score_test_mean

def Stratifed(X,Y):
    {

    }



x_val_df = pd.DataFrame(data=x_val, columns= features_name)
x_test_df = pd.DataFrame(data=x_test, columns= features_name)

model = LinearSVC().fit(x_train, y_train)
score = model.score(x_test, y_test)
t=3
#x_test_df["session_length"] = 


h=34


"""
kf = sklearn.cross_validation.KFold(4, n_folds=2)
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
"""

