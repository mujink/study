
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

# 딕셔너리 키벨류 리스트 형식
parameters = [
    {"C" : [1, 10, 100, 1000], "kernel" : ["linear"]},
    {"C" : [1, 10, 100], "kernel" : ["rbf"], "gamma" : [0.001,0.0001]    },
    {"C" : [1, 10, 100, 1000], "kernel" : ["sigmoid"], "gamma" : [0.001, 0.000,]}
]

# 모델, 파라미터, 교차검증
model = GridSearchCV(SVC(), parameters, cv=kfold)
score = cross_val_score(model, x_train, y_train, cv=kfold)

print("교차검증 점수", score)

"""
time 0:00:00.088791
SVC(C=1, kernel='linear')
최종 정답률 : 1.0
score 1.0
"""