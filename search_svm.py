
import pandas as pd
import numpy
import time
from pandas._config import config
from ray import tune
from scipy.stats.stats import sigmaclip
import sklearn.datasets as skl
from ray.tune.suggest.bayesopt import BayesOptSearch
from sklearn.model_selection import cross_val_score,train_test_split
import datetime
from sklearn.svm import SVC
from xgboost import XGBClassifier
import os

from scipy.spatial.distance import cdist

############################ Grid search de base ########################

############################### 1. Grid XGBClassifier

## Si prend trop de temps réduire cross val 
def objective_SVC(X_train,y_train,C,gamma):
    """The function that evaluate the combination of parameter. 

    Returns:
        [type]: Score on the X_validation after train on the X_train.
    """
    # Split the dataset
    X_train_2,X_valid,y_train_2,y_valid = train_test_split(X_train,y_train,stratify=y_train,test_size=0.5)
    # Define the model
    model = SVC(C=C,gamma=gamma)
    model.fit(X_train_2,y_train_2)
    # Evaluate
    score = model.score(X_valid,y_valid)
    
    return score

def training_function(config):
    """Call the objective function and report the score to tune

    Args:
        config ([type]): [description]
    """
    # Hyperparameters
    #"reg_lambda":5, #tune.grid_search(list(reg_lambda)),
    #        "subsample":0.8,#tune.grid_search(list(subsample)),
    #        "colsample_bytree":0.8, #tune.grid_search(list(colsample_bytree)),
    #        "scale_pos_weight":3.0, #tune.grid_search(list(scale_pos_weight)),
    #        "resources_per_trial":{"CPU":1}
    C,gamma = config["C"],config["gamma"]
    X_train,y_train = config["X_train"], config["y_train"]
    intermediate_score = objective_SVC(X_train,y_train,C,gamma)
    # Feed the score back back to Tune.
    tune.report(score=intermediate_score)

def calculate_sigma_min(X_train):
    x = cdist(X_train,X_train)

    mask = numpy.ones(x.shape,dtype=bool)
    numpy.fill_diagonal(mask,0)
    sigma_min = x[mask].min()
    return sigma_min
def MinEuclideanDist(matrix):
    minDist = 9999
    for i, a in enumerate(matrix):
        for j, b in enumerate(matrix):
            if i >= j:
                continue
            
            dist = numpy.linalg.norm(a-b)
            
            if dist < minDist:
                minDist = dist
        
    return minDist

def search_grid_SVC(X_train,y_train,result_folder_path,sigma_min):
    """Find the best parameter for the XGB method. Save all the configuration

    Args:
        X_train ([type]): [description]
        y_train ([type]): [description]

    Returns:
        [dict]: Best parameter of the XGB
    """
    # Calculate gamma parameter
    #start = time.time()
    
    list_sigma = numpy.linspace(1,64,3)*sigma_min # 20
    #end = time.time()
    #print("Cdist time time : ",end-start)
    print("sigma_min",sigma_min)
    # Nos paramètre à optimiser
    C = numpy.array(numpy.logspace(-5,5,4)) #11
    gamma = 0.5*list_sigma**(-2)#numpy.array(numpy.arange(2,8,2))
    

    # La commande qui lance la recherche sur nos parmètres
    analysis = tune.run(
        training_function,
        config={'C': tune.grid_search(list(C)),
        'gamma': tune.grid_search(list(gamma)),
        "X_train":X_train,
        "y_train":y_train
        },
        verbose=0)
    
    best_config_r = analysis.get_best_config(metric="score", mode="max")
    best_config_r.pop("X_train")
    best_config_r.pop("y_train")
    print("Best config: ", best_config_r)
    results = []
    i = 0 
    all_config_score= analysis.results
    for result in all_config_score.values():
        config = result["config"]
        config.pop("X_train")
        config.pop("y_train")
        config["total_time"] = result["time_total_s"]
        config["score"] = result["score"]
        results.append(config)
    
    #all_config_score.pop("X_train")
    #all_config_score.pop("y_train")
    df_result = pd.DataFrame.from_records(results)
    path_ = os.path.join(result_folder_path,"SVC_grid_search_scores.csv")
    df_result.to_csv(path_)
    
    best_config_r["method"] = "search_grid_svc"

    return best_config_r


def evaluate_SVC(X_test,y_test,X_train,y_train,best_config_r,result_path):
    """Calculate the score of the train set and the test set for the best parameter configuration.

    Args:
        X_test ([type]): [description]
        y_test ([type]): [description]
        X_train ([type]): [description]
        y_train ([type]): [description]
        best_config_r ([type]): [description]

    Returns:
        [list]: Average test_scores, the different test_score, train_score
    """
   
    C, gamma = best_config_r["C"], best_config_r["gamma"]
    
    # Calculate the score of the test 
    scores = cross_val_score(SVC(C=C,gamma=gamma), X_test, y_test,cv=5)
    average_score = numpy.mean(scores)
    # Claculate the training score
    model = SVC(C=C,gamma=gamma)
    model.fit(X_train,y_train)
    score_train = model.score(X_train,y_train)

    # save score
    best_config_r["average_score_test"] = average_score
    best_config_r["score_train"] = score_train
    results_best_config = best_config_r
    poids_variable = model.coef_
    
    df_result = pd.DataFrame(results_best_config,index=[0])
    path_ = os.path.join(result_path,"XGB_grid_search_best_config.csv")
    df_result.to_csv(path_)
    

    return average_score,scores,results_best_config