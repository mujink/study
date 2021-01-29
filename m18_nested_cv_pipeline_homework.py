# 모델 랜덤포레스트 쓰고
# 파이프라인 엮어서 25번 돌리기!
# 데이터는 디아벳

import numpy as np
from sklearn.datasets import load_diabetes

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline

import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()

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
            # make_pipeline 은  키워드 없는 형태의 인풋을 Pipeline 형식으로 키워드를 입혀 Pipeline에 인풋 값을 리턴하는 함수임
            # 만약 키워드가 있으면 그대로 Pipeline에 반환하기 때문에 잘 돌아감. 
            pipe = pip([("scaler", pro()),("Rand", RandomForestRegressor())])
        for Search in model:
            print("서치 : ",Search)
            models = Search(pipe,parameters, cv=kfold )
            score = cross_val_score(models, x, y, cv=kfold)
            print("교차검증 점수", score)
            


# 교차 검증이 총 8번 나와야함 아래 포문을 돌렸음
# pips = [Pipeline, make_pipeline]
# proprecess = [MinMaxScaler, StandardScaler]
# model = [RandomizedSearchCV, GridSearchCV]
"""
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.55406586 0.34332455 0.48902554 0.40200653 0.41921902]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.47578207 0.41603688 0.40130917 0.46566179 0.38323501]
pip : <function make_pipeline at 0x0000028FC107B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.530552   0.41130991 0.38308633 0.31209303 0.51339556]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.47490152 0.32387272 0.52327724 0.44474764 0.51745256]
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.53581849 0.48027771 0.4273793  0.48260655 0.23721432]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.50134945 0.4278494  0.27712109 0.40101549 0.44045034]
pip : <function make_pipeline at 0x0000028FC107B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
교차검증 점수 [0.44962403 0.28730232 0.46215891 0.45022282 0.31531396]
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
교차검증 점수 [0.49461075 0.36726983 0.55511256 0.43126985 0.3812764 ]
"""