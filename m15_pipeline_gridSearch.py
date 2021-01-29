# randomforest config

import numpy as np
from sklearn.datasets import load_iris

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score

# 머신러닝의 분류 모델들
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # 회귀아님

# data + preprocessing Auto= pipeline
from sklearn.pipeline import Pipeline, make_pipeline

import warnings

warnings.filterwarnings('ignore')

# data
dataset = load_iris()

x = dataset.data
y = dataset.target
print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# model
# 파이프라인 입력은 스탭스= 리스트, 튜플, 키워드 형식으로 작성한다.
# model = Pipeline([("Minmax", MinMaxScaler()), ('model', SVC())])

# 파이프라인 입력은 스탭스= 리스트 형식으로 작성한다.
parameters = [
    {"svc__C" : [1, 10, 100, 1000], "svc__kernel" : ["linear"]},
    {"svc__C" : [1, 10, 100], "svc__kernel" : ["rbf"], "svc__gamma" : [0.001,0.0001]    },
    {"svc__C" : [1, 10, 100, 1000], "svc__kernel" : ["sigmoid"], "svc__gamma" : [0.001, 0.000,]}
]

# parameters = [
#     {"mal__C" : [1, 10, 100, 1000], "mal__kernel" : ["linear"]},
#     {"mal__C" : [1, 10, 100], "mal__kernel" : ["rbf"], "mal__gamma" : [0.001,0.0001]    },
#     {"mal__C" : [1, 10, 100, 1000], "mal__kernel" : ["sigmoid"], "mal__gamma" : [0.001, 0.000,]}
# ]

# 어플리케이션은 인풋, 클라스, 메소드, 함수를 속성으로 얽혀 인터페이스로 사이킷런 및 다른 라이브러리 사이의 입출력을 제어하는 구성으로 되어있음
# 어플리케이션의 입력은 내부 인터페이스를 통해 초기화되어 출력되고
# 출력되는 값들이 다음 어플리케이션의 인풋으로 입력된다.
# 어플리케이션에서 출력되는 값들을 받아 사용하는 다음 어플리케이션 라이브러리의 인풋으로 받을 때, 다음 어플리케이션의 인터페이스에
# 이전 어플리케이션의 출력의 처리가 명시되어 있으면 오류없이 정상 동작한다.
# 직관적인 작성은 가능하나 세부사항을 확인하기 어렵고, 다양한 기능을 빠르게 조합하고 구현 할 수 있다.
pipe = make_pipeline(MinMaxScaler(), SVC())
# pipe = Pipeline([("scaler", MinMaxScaler()),("svc", SVC())])
print(pipe)

models = RandomizedSearchCV(pipe,parameters, cv=5 )
# models = GridSearchCV(pipe,parameters, cv=5 )
models.fit(x_train,y_train)
results = models.score(x_test,y_test)
print("results :", results)
"""
(150, 4) (150,)
results : 0.9666666666666667

(150, 4) (150,)
results : 0.9666666666666667
"""
