import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Loader les données nécessaire
X = pd.read_csv('features_te.csv').to_numpy()
y = pd.read_csv('target.csv').to_numpy()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.50,stratify=y,random_state=1)
X_train_2,X_valid,y_train_2,y_valid = train_test_split(X_train,y_train,stratify=y_train,test_size=0.5,random_state=1)

# XGB
n_estimator = 1
max_depth = 1
depth = 1
l_rate = 1
reg_lambda = 1
subsample = 1
colsample_bytree = 1
scale_pos_weight = 1

# Créer le modèle et l'entraîner
model_xgb = XGBClassifier(verbosity=0,silent = True,use_label_encoder=False,n_estimators=n_estimator, max_depth=depth,learning_rate=l_rate,reg_lambda=reg_lambda,subsample=subsample,colsample_bytree=colsample_bytree,scale_pos_weight=scale_pos_weight)
model_xgb.fit(X_train_2, y_train_2)

# Score d'entrainement
y_predict_train = model_xgb.predict(X_train_2)
score_train = accuracy_score(y_train_2, y_predict_train)

# Score de test
y_predict_test = model_xgb.predict(X_test)
score_test = accuracy_score(y_test, y_predict_test)

# Matrice de confusion
confusion_matrix_test = confusion_matrix(y_test, y_predict_test)

# SVC
C = 1
gamma = 1
kernel = 1

# Créer le modèle et l'entraîner
model_xgb = model = SVC(C=C,gamma=gamma,kernel=kernel)
model_xgb.fit(X_train_2, y_train_2)

# Score d'entrainement
y_predict_train = model_xgb.predict(X_train_2)
score_train = accuracy_score(y_train_2, y_predict_train)

# Score de test
y_predict_test = model_xgb.predict(X_test)
score_test = accuracy_score(y_test, y_predict_test)

# Matrice de confusion
confusion_matrix_test = confusion_matrix(y_test, y_predict_test)