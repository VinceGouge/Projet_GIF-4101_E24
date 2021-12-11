import pandas as pd
from sklearn.model_selection import train_test_split
import time
import sklearn.datasets  as skl
import search_algo as sal
import search_svm as ss
# 1. Load the data 
start= time.time()

#x = pd.read_csv('features_te.csv').to_numpy()
#y = pd.read_csv('target.csv').to_numpy()


# 2. Crée un jeu de test et un jeu de train 

#X_train, X_test,y_train,y_test = train_test_split(x,y,stratify=y,test_size=0.5)

########## pour tester 
X, y= skl.make_moons(n_samples=10000, shuffle=True, noise=True, random_state=None)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.50,stratify=y)
print(X.shape)
# 3. test préliminaire des classifieurs
#   3.1 Classifieurs 1 (SVC linear)
        # Calculer la valeur de sigma min

#       3.1.1 Methode search 1 (Max-Max)
#       3.1.2 Methode search 2 (Grid search)
# 
#   3.2 Classifieurs 2 (XGBoost)
#       3.2.1 Methode search 1
# Faire graphique RFE 
# Y score, x RFE
# 
# Le run
# Tableau des meilleurs scores 
# 




path = sal.get_folder_path()
best_config = ss.search_grid_SVC(X_train,y_train,path)

#best_config = sal.search_grid_XGB_2(X_train,y_train,path)
print(best_config)
#sal.evaluate_XGB(X_test,y_test,X_train,y_train,best_config,path)

        