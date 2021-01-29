# 랜덤서치와 그리드서치, 파이프라인을 엮어라
# 모델은 렌덤포레스트

import numpy as np
from sklearn.datasets import load_iris

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score

# 머신러닝의 분류 모델들
# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression # 회귀아님

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

# 파이프라인 입력은 스탭스= 리스트 형식으로 작성한다.
parameters = [
    {"Rand__n_estimators" : [1, 10, 100], "Rand__min_samples_split" : [6,8,10]},
    {"Rand__n_estimators" : [2, 200], "Rand__min_samples_split" : [2,4,6], "Rand__max_depth" : [10,100]    },
    {"Rand__n_estimators" : [3, 10, 300], "Rand__min_samples_split" : [12,14,16], "Rand__criterion" : ["gini","entropy"]}
]


proprecess = [MinMaxScaler, StandardScaler]
pips = [Pipeline, make_pipeline]
model = [RandomizedSearchCV, GridSearchCV]

for pro in proprecess:
    for pip in pips:
        print("pip :",pip)
        # make_pipeline 은 <function make_pipeline at 0x00000286DE3C5310> 로 확인됨
        # make_pipeline 은 Pipeline 로 인식함
        if pip == Pipeline:
            # make_pipeline 은  키워드 없는 형태의 인풋을 Pipeline 형식으로 Pipeline에 인풋 값을 리턴하기 때문에 잘 돌아감. 
            pipe = pip([("scaler", pro()),("Rand", RandomForestClassifier())])
        for Search in model:
            print("서치 : ",Search)
            models = Search(pipe,parameters, cv=5 )
            models.fit(x_train,y_train)
            results = models.score(x_test,y_test)
            print("results :", results)

                
"""
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
results : 0.9666666666666667
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
results : 0.9666666666666667
pip : <function make_pipeline at 0x000001FCBE1853A0>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
results : 0.9666666666666667
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
results : 0.9666666666666667
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
results : 0.9333333333333333
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
results : 0.9666666666666667
pip : <function make_pipeline at 0x000001FCBE1853A0>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
results : 0.9666666666666667
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
results : 0.9333333333333333
"""
