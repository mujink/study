import numpy as np
from sklearn.datasets import load_boston

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score

# 머신러닝의 분류 모델들
# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression # 회귀아님

# # 머신러닝 회귀 모델들
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import  KNeighborsRegressor
# from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# import warnings

import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"n_estimators" : [1, 10, 100], "min_samples_split" : [6,8,10]},
    {"n_estimators" : [2, 200], "min_samples_split" : [2,4,6], "max_depth" : [10,100]    },
    {"n_estimators" : [3, 10, 300], "min_samples_split" : [12,14,16], "criterion" : ["gini","entropy"]}
]


model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold)

import datetime

start = datetime.datetime.now()
model.fit(x_train,y_train)
end = datetime.datetime.now()
print("time", end-start)

print(model.best_estimator_)

y_pred = model.predict(x_test)
print("최종 정답률 :", r2_score(y_test, y_pred))

aaa =model.score(x_test,y_test)
print("score", aaa)

"""
time 0:00:14.383580
RandomForestRegressor(max_depth=10, n_estimators=200)
최종 정답률 : 0.9112324228097607
score 0.9112324228097607
"""