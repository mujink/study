# 모델 포문으로 훈련, 평가하기

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

# 머신러닝의 분류 모델들
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # 분류임 회귀아님

import warnings

warnings.filterwarnings('ignore')

datasets = load_iris()
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
models = [LinearSVC, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression]

for algorithm in models :
    model = algorithm()
    print(algorithm)
    scores = cross_val_score(model, x_train, y_train, cv= kfold)
    print('scores : ', scores)


"""
<class 'sklearn.svm._classes.LinearSVC'>
scores :  [0.91666667 0.95833333 0.91666667 1.         0.95833333]
<class 'sklearn.svm._classes.SVC'>
scores :  [0.91666667 1.         1.         1.         0.91666667]
<class 'sklearn.neighbors._classification.KNeighborsClassifier'>
scores :  [0.91666667 0.95833333 1.         1.         0.91666667]
<class 'sklearn.tree._classes.DecisionTreeClassifier'>
scores :  [0.95833333 0.95833333 0.875      0.95833333 1.        ]
<class 'sklearn.ensemble._forest.RandomForestClassifier'>
scores :  [0.95833333 0.95833333 0.95833333 0.91666667 1.        ]
<class 'sklearn.linear_model._logistic.LogisticRegression'>
scores :  [1.         1.         0.95833333 0.91666667 0.95833333]
"""
