from ray import tune

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch

import pandas as pd
import numpy 
# Import a dataset 
url = 'https://media.githubusercontent.com/media/PacktPublishing/Hands-On-Gradient-Boosting-with-XGBoost-and-Scikit-learn/master/Chapter02/heart_disease.csv'
df = pd.read_csv(url)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

## Define the objectives valeur ( la fonction à optimiser)
def objective(step, alpha, beta):
    return (0.1 + alpha * step / 100)**(-1) + beta * 0.1

def objective_fct(X,y,n_estimator,depth,learning_rate,reg_lambda,subsample,colsample_bytree,scale_pos_weight):
    pass
    

def training_function(config):
    #clf = XGBClassifier(verbosity=0,silent = True,use_label_encoder=False)
    #clf.fit(X,y)
    #scores = cross_val_score(XGBClassifier(verbosity=0,silent = True,use_label_encoder=False,n_estimators=config["n_estimator"],max_depth=config["depth"],learning_rate=config["learning_rate"],reg_lambda=config["reg_lambda"],subsample=config["subsample"],colsample_bytree=config["colsample_bytree"],scale_pos_weight=config["scale_pos_weight"]), X, y)
    
    scores = cross_val_score(XGBClassifier(verbosity=0,silent = True,use_label_encoder=False,n_estimators=config["n_estimator"],max_depth=config["depth"],learning_rate=0.53,reg_lambda=6,subsample=1,colsample_bytree=0.4,scale_pos_weight=1), X, y)
        
    average_score = numpy.mean(scores)
    tune.report(score=average_score)
config = {"n_estimator":tune.quniform(10,510,10),
"depth":tune.quniform(2,22,2)}
#,
#"learning_rate":tune.quniform(0.01,10,0.01),
#"reg_lambda":tune.quniform(1,11,1),
#"subsample":tune.quniform(0.2,1.2,0.2),
#"colsample_bytree":tune.quniform(0.2,1.2,0.2),
#"scale_pos_weight":tune.quniform(1,4.5,0.5)
#}

analysis = tune.run(
    training_function,
    config=config,
    metric="score",
    mode="max",
    search_alg=ConcurrencyLimiter(
        BayesOptSearch(random_search_steps=4),
        max_concurrent=2),
    num_samples=20,
    stop={"training_iteration": 20},
    verbose=2)

print("Best config: ", analysis.get_best_config(
    metric="mean_loss", mode="min"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df