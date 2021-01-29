# 모델 랜덤포레스트 쓰고
# 파이프라인 엮어서 25번 돌리기!
# 데이터는 와인


import numpy as np
from sklearn.datasets import load_wine

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline

import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()

x = dataset.data
y = dataset.target

kfold = KFold(n_splits=5, shuffle=True)

# 딕셔너리 키벨류 리스트 형식
parameters = [
    {"Rand__n_estimators" : [1, 10, 100], "Rand__min_samples_split" : [6,8,10]},
    {"Rand__n_estimators" : [2, 200], "Rand__min_samples_split" : [2,4,6], "Rand__max_depth" : [10,100]    },
    {"Rand__n_estimators" : [3, 10, 300], "Rand__min_samples_split" : [12,14,16], "Rand__criterion" : ["gini","entropy"]}
]

pips = [Pipeline, make_pipeline]
proprecess = [MinMaxScaler, StandardScaler]
model = [RandomizedSearchCV, GridSearchCV]

# 모델, 파라미터, 교차검증

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
            models = Search(pipe,parameters, cv=kfold )
            score = cross_val_score(models, x, y, cv=kfold)
            print("교차검증 점수", score)
            

# 교차 검증이 총 8번 나와야함 아래 포문을 돌렸음
# pips = [Pipeline, make_pipeline] 파이프라인 2 셋
# proprecess = [MinMaxScaler, StandardScaler] 전처리 2셋
# model = [RandomizedSearchCV, GridSearchCV] 파이프라인 2셋
"""
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [1.         0.91666667 0.91666667 1.         1.        ]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.86111111 1.         0.94444444 0.97142857 1.        ]
pip : <function make_pipeline at 0x000001E82696B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.94444444 1.         0.97222222 1.         0.94285714]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [1.         0.97222222 0.97222222 0.97142857 0.91428571]
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.94444444 0.97222222 1.         1.         0.97142857]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.94444444 1.         1.         0.97142857 1.        ]
pip : <function make_pipeline at 0x000001E82696B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.91666667 1.         1.         1.         0.97142857]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.97222222 0.83333333 1.         1.         0.97142857]
"""