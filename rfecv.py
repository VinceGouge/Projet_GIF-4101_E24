from yellowbrick.model_selection import RFECV
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Create a dataset with only 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3, n_redundant=2, n_repeated=0, n_classes=8, n_clusters_per_class=1, random_state=0)

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