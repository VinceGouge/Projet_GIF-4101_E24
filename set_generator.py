from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

X = pd.read_csv('features.csv').to_numpy()
y = pd.read_csv('target.csv').to_numpy()

X_optimization, X_test, y_optimization, y_test = train_test_split(X,y,stratify=y, test_size=0.5)

