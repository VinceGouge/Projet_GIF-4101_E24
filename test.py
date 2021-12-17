from re import X
from numpy import linalg as LA
import numpy
from scipy.spatial.distance import cdist
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
X = pd.read_csv('features_te.csv').to_numpy()
y = pd.read_csv('target.csv').to_numpy()


# 2. Crée un jeu de test et un jeu de train 

#X_train, X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.5)
########## pour tester 
#X, y= skl.make_moons(n_samples=1000, shuffle=True, noise=False, random_state=None)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.50,stratify=y)

X_train_2,X_valid,y_train_2,y_valid = train_test_split(X_train,y_train,test_size=0.50,stratify=y_train)






model = XGBClassifier(verbosity=0,silent = True,use_label_encoder=False,n_estimators=260,
    max_depth=2,learning_rate=1.023292992280754,reg_lambda=6,subsample=1.0,colsample_bytree=0.6,scale_pos_weight=1.0)

clf_valid = model.fit(X_train_2,y_train_2)
score_valid = clf_valid.score(X_valid,y_valid)
print("SCORE VALID",score_valid)
print("alllko")

clf = model.fit(X_train,y_train)
print("Time 2 predict")
predictions = clf.predict(X_test)
print(predictions)
print(y_train)

matrix_confusius = confusion_matrix(y_test,predictions)
print("_________________ Confusion matrix")
print(matrix_confusius)
print("_________________ Score")
score_valid = clf_valid.score(X_train,y_train)
print("SCORE VALID",score_valid)
score = clf.score(X_test,y_test)

print(score)


for i in range(3):
    print("________________________________\n")
c = 1e-5
gamma = 59525776 
kernel= "rbf"
model = SVC(C=c,gamma=gamma,kernel=kernel)
model.fit(X_train,numpy.ravel(y_train))

score_train = model.score(X_train,y_train)
print("score_train:", score_train)
predictions = clf.predict(X_test)
score_test = model.score(X_test,y_test)

matrix_confusius = confusion_matrix(y_test,predictions)
print("score_test:", score_test)