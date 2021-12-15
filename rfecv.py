from re import T
import numpy as np

from yellowbrick.model_selection import RFECV
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris

# Create a dataset with only 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3, n_redundant=2, n_repeated=0, n_classes=8, n_clusters_per_class=1, random_state=0)
#X, y = load_iris(return_X_y=True)

# Instancier la cross validation avec KFold = 5
cross_validation = StratifiedKFold(5)

# Instancier le model avec paramètres optimaux
model = SVC(kernel='linear', C=1)

# Instancier le RFECV avec le model voulut et la cross validation déclarée
visualizer = RFECV(model, cv=cross_validation)

# Fit du modèle sur les données
visualizer.fit(X, y)

# Afficher le graph
visualizer.show()

# Print l'ordre
ranking = (visualizer.ranking_)
nom_features = ["f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22","f23","f24"]

index_list = []
value_list = []

for index, value in enumerate(ranking):
    index_list.append(index)
    value_list.append(value)
ranking_zip = zip(nom_features, value_list)
ranking_sorted = sorted(ranking_zip, key=lambda tup:tup[1])

for x in ranking_sorted:
    print('Le feature ' + str(x[0]) + ' est classé : ' + str(x[1]))