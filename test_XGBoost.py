
import pandas as pd
import numpy
import time
from ray import tune


url = 'https://media.githubusercontent.com/media/PacktPublishing/Hands-On-Gradient-Boosting-with-XGBoost-and-Scikit-learn/master/Chapter02/heart_disease.csv'
df = pd.read_csv(url)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV

#n_estimators : number of boosting rounds (20)
#Max_depth : The maximum tree detph
#learning_rate: boosting learning rate
#reg_lambda : reg L2
# subsample : subsample ratio of the training instance
# colsample_bytree : ratio of columns when contructing each tree.  
# scale_pos_weight : Balancing of positiv and negative weight

n_estimators = numpy.array(numpy.arange(10,30,10)) #510
max_depth= numpy.array(numpy.arange(2,8,2))
learning_rate=numpy.array(numpy.logspace(0.01,0.8,10))
reg_lambda=numpy.array(numpy.arange(1,11,1))
subsample=numpy.array(numpy.arange(0.2,1.2,0.2))
colsample_bytree=numpy.array(numpy.arange(0.2,1.2,0.2))
scale_pos_weight=numpy.array(numpy.arange(1,4.5,0.5))

param_grid = {"n_estimators":n_estimators,"max_depth":max_depth,"learning_rate":learning_rate,
"reg_lambda":reg_lambda,"subsample":subsample,"colsample_bytree":colsample_bytree,"scale_pos_weight":scale_pos_weight,"verbosity":[0],"silent":[True]}

#def test_param(X,y,n_estimators,max_depth,learning_rate,reg_lambda,subsample,colsample_bytree,scale_pos_weight:
dict_scores = {}
i= 0 

start = time.time()
grid_search = GridSearchCV(XGBClassifier(verbosity=0,silent = True,use_label_encoder=False),param_grid=param_grid)
grid_search.fit(X,y)

delta_time = time.time()-start
print(f"total time {delta_time} seconds")
print(f"total time {delta_time/i} seconds/simulation")
for n_estimator in n_estimators:
    for  depth in max_depth:
        scores = cross_val_score(XGBClassifier(verbosity=0,silent = True,use_label_encoder=False,n_estimators=n_estimator,max_depth=depth,learning_rate=0.53,reg_lambda=6,subsample=1,colsample_bytree=0.4,scale_pos_weight=1), X, y)
        print(scores)
        average_score = numpy.mean(scores)
        print(f"{i} n_estimaor {n_estimator} depth {depth} score: {average_score} ")
        #dict_scores[i)] = average_score
        i += 1
delta_time = time.time()-start
print(f"total time {delta_time} seconds")
print(f"total time {delta_time/i} seconds/simulation")