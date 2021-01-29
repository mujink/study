# randomforest config

import numpy as np
from sklearn.datasets import load_breast_cancer

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
dataset = load_breast_cancer()

x = dataset.data
y = dataset.target
print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# model
# 파이프라인 입력은 스탭스= 리스트, 튜플, 키워드 형식으로 작성한다.
# model = Pipeline([("Minmax", MinMaxScaler()), ('model', SVC())])

# 파이프라인 입력은 스탭스= 리스트 형식으로 작성한다.
# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
models = [LinearSVC, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression]
proprecess = [MinMaxScaler, StandardScaler]

for algorithm in models :
    model = algorithm()
    for process in proprecess :
        process = process()
        models = make_pipeline(process, model)
        print("전처리 : ", process, "모델 :", model)
        models.fit(x_train,y_train)
        results = models.score(x_test,y_test)
        print("results :", results)

"""
전처리 :  MinMaxScaler() 모델 : LinearSVC()
results : 0.9736842105263158
전처리 :  StandardScaler() 모델 : LinearSVC()
results : 0.9736842105263158
전처리 :  MinMaxScaler() 모델 : SVC()
results : 0.9736842105263158
전처리 :  StandardScaler() 모델 : SVC()
results : 0.9736842105263158
전처리 :  MinMaxScaler() 모델 : KNeighborsClassifier()
results : 0.9473684210526315
전처리 :  StandardScaler() 모델 : KNeighborsClassifier()
results : 0.956140350877193
전처리 :  MinMaxScaler() 모델 : DecisionTreeClassifier()
results : 0.9385964912280702
전처리 :  StandardScaler() 모델 : DecisionTreeClassifier()
results : 0.956140350877193
전처리 :  MinMaxScaler() 모델 : RandomForestClassifier()
results : 0.9473684210526315
전처리 :  StandardScaler() 모델 : RandomForestClassifier()
results : 0.956140350877193
전처리 :  MinMaxScaler() 모델 : LogisticRegression()
results : 0.956140350877193
전처리 :  StandardScaler() 모델 : LogisticRegression()
results : 0.9736842105263158
"""
