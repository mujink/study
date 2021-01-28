import numpy as np
from sklearn.datasets import load_iris

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC

import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"C" : [1, 10, 100, 1000], "kernel" : ["linear"]},
    {"C" : [1, 10, 100], "kernel" : ["rbf"], "gamma" : [0.001,0.0001]    },
    {"C" : [1, 10, 100, 1000], "kernel" : ["sigmoid"], "gamma" : [0.001, 0.000,]}
]

model = GridSearchCV(SVC(), parameters, cv=kfold)

import datetime

start = datetime.datetime.now()
model.fit(x_train,y_train)
end = datetime.datetime.now()
print("time", end-start)

print(model.best_estimator_)

y_pred = model.predict(x_test)
print("??", accuracy_score(y_test, y_pred))

aaa =model.score(x_test,y_test)
print("score", aaa)

"""
time 0:00:00.088791
SVC(C=1, kernel='linear')
?? 1.0
score 1.0
"""