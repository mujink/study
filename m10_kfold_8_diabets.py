# 모델 포문으로 훈련, 평가하기

import numpy as np
from sklearn.datasets import load_diabetes
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

datasets = load_diabetes()
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
scores :  [0.57282107 0.37412966 0.54031696 0.33943766 0.5944223 ]
<class 'sklearn.neighbors._regression.KNeighborsRegressor'>
scores :  [0.56524311 0.36022565 0.32144867 0.11645496 0.39092077]
<class 'sklearn.tree._classes.DecisionTreeRegressor'>
scores :  [-0.04054532 -0.06343199  0.14450489  0.39155325  0.35151108]
<class 'sklearn.ensemble._forest.RandomForestRegressor'>
scores :  [0.47835573 0.42786651 0.22316319 0.44410309 0.50698946]
"""
