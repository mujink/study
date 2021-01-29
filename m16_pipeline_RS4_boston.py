# 랜덤서치와 그리드서치, 파이프라인을 엮어라
# 모델은 렌덤포레스트

import numpy as np
from sklearn.datasets import load_boston

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score

# # 머신러닝 회귀 모델들
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import  KNeighborsRegressor
# from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# data + preprocessing Auto= pipeline
from sklearn.pipeline import Pipeline, make_pipeline

import warnings

warnings.filterwarnings('ignore')

# data
dataset = load_boston()

x = dataset.data
y = dataset.target
print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# model
# 파이프라인 입력은 스탭스= 리스트, 튜플, 키워드 형식으로 작성한다.
# model = Pipeline([("Minmax", MinMaxScaler()), ('model', SVC())])

# 파이프라인 입력은 스탭스= 리스트 형식으로 작성한다.
parameters = [
    {"Rand__n_estimators" : [1, 10, 100], "Rand__min_samples_split" : [6,8,10]},
    {"Rand__n_estimators" : [2, 200], "Rand__min_samples_split" : [2,4,6], "Rand__max_depth" : [10,100]    },
    {"Rand__n_estimators" : [3, 10, 300], "Rand__min_samples_split" : [12,14,16], "Rand__criterion" : ["gini","entropy"]}
]


# make_pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
# make_pipe = make_pipeline
# pipe = Pipeline([("scaler", MinMaxScaler()),("Rand", RandomForestClassifier())])
# models = RandomizedSearchCV(pipe,parameters, cv=5 )
# models = GridSearchCV(pipe,parameters, cv=5 )

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
            pipe = pip([("scaler", pro()),("Rand", RandomForestRegressor())])
        for Search in model:
            print("서치 : ",Search)
            models = Search(pipe,parameters, cv=5 )
            models.fit(x_train,y_train)
            results = models.score(x_test,y_test)
            print("results :", results)

                
"""
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
results : 0.9118899922029775
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
results : 0.9124965807119697
pip : <function make_pipeline at 0x0000024F1AC6B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
results : 0.9130580400948217
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
results : 0.91102745973482
pip : <class 'sklearn.pipeline.Pipeline'>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
results : 0.9092209498810044
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
results : 0.9123097729293295
pip : <function make_pipeline at 0x0000024F1AC6B310>
서치 :  <class 'sklearn.model_selection._search.RandomizedSearchCV'>
results : 0.9114815662216257
서치 :  <class 'sklearn.model_selection._search.GridSearchCV'>
results : 0.9158477741444818
"""
