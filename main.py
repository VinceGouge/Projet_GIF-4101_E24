import pandas as pd
from sklearn.model_selection import train_test_split
import time
import sklearn.datasets  as skl
import search_algo as sal
import search_svm as ss
import os.path as osp
# 1. Load the data 
start= time.time()

X = pd.read_csv('features_te.csv').to_numpy()
y = pd.read_csv('target.csv').to_numpy()


# 2. Crée un jeu de test et un jeu de train 

########## pour tester 
#X, y= skl.make_moons(n_samples=1000, shuffle=True, noise=False, random_state=None)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.50,stratify=y,random_state=1)


# 2.5 Créer le fichier de résultats
path = sal.get_folder_path(name="Test")

best_score_summary = []
# 3. test préliminaire des classifieurs
#   3.1 Classifieurs 1 (SVC linear)
        # Calculer la valeur de sigma min
sigma_min = 2.82e-6
#best_config = ss.search_grid_SVC(X_train,y_train,path,sigma_min)
#print(best_config)
#       3.1.1 Methode search 1 (Max-Max)
#       3.1.2 Methode search 2 (Grid search)
#best_config_parameter_search_grid_ss = ss.search_grid_SVC(X_train,y_train,path,sigma_min=sigma_min)
#best_config_result_score_svc_search_grid = ss.evaluate_SVC(X_test,y_test,X_train,y_train,best_config_parameter_search_grid_ss,path)
#best_score_summary.append(best_config_result_score_svc_search_grid)

time_svc = round(time.time()-start,2)
print("_________________________\n \n","SVC COMPLETEEEEEEED in", time_svc," s \n_________________________\n \n")
#svm_best_score = ss.evaluate_SVC(X_test,y_test,X_train,y_train,best_config,path)
#   3.2 Classifieurs 2 (XGBoost)

#       3.2.1 Methode search 1
#best_config_parameter_max = sal.search_grid_XGB_max_max(X_train,y_train,path)

#best_config_result_score_XGB_max_max = sal.evaluate_XGB(X_test,y_test,X_train,y_train,best_config_parameter_max,path)
#best_score_summary.append(best_config_result_score_XGB_max_max)

#       Estimate the  Search method
best_config_parameter_search_grid = sal.search_grid_XGB_max_max_boucler(X_train,y_train,path, nombre_iteration_=3)

best_config_result_score_XGB_search_grid = sal.evaluate_XGB(X_test,y_test,X_train,y_train,best_config_parameter_search_grid,path)
best_score_summary.append(best_config_result_score_XGB_search_grid)
time_XGB = round(time.time()-start-time_svc,2)
print("_________________________\n \n","XGB COMPLETEEEEEEED in", time_XGB ,"s \n_________________________\n \n")





summary_csv = pd.DataFrame.from_records(best_score_summary)
summary_csv.to_csv(osp.join(path,"summary.csv"))

print("Total time",round(time.time()-start))
# Faire graphique RFE 
# Y score, x RFE
# 
# Le run
# Tableau des meilleurs scores 
# 

        