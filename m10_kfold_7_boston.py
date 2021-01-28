# 모델 포문으로 훈련, 평가하기

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

# # 머신러닝의 분류 모델들
# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression # 회귀아님

# 머신러닝 회귀 모델들
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

datasets = load_boston()
x = datasets.data
y = datasets.target
# print(datasets.DESCR)
# print(datasets.feature_names)
# print(datasets.target_names)


print(x.shape)      #(150,4)
print(y.shape)      #(150,3)

#1.1 Data Preprocessing / KFold

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=1)

kfold = KFold(n_splits=5, shuffle=True)

#2.model
models = [LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]

for algorithm in models :
    model = algorithm()
    print(algorithm)
    scores = cross_val_score(model, x_train, y_train, cv= kfold)
    print('scores : ', scores)


"""
<class 'sklearn.linear_model._base.LinearRegression'>
scores :  [0.77596882 0.67483953 0.72252283 0.56359232 0.81675334]
<class 'sklearn.neighbors._regression.KNeighborsRegressor'>
scores :  [0.517859   0.44329749 0.45433054 0.4682997  0.49022234]
<class 'sklearn.tree._classes.DecisionTreeRegressor'>
scores :  [0.85615298 0.8189722  0.7746489  0.68845536 0.75648656]
<class 'sklearn.ensemble._forest.RandomForestRegressor'>
scores :  [0.91078065 0.76528193 0.90825741 0.86780062 0.8668954 ]
"""
