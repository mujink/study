# randomforest config

import numpy as np
from sklearn.datasets import load_diabetes

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score

# 머신러닝의 분류 모델들
# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression # 회귀아님


# # 머신러닝 회귀 모델들
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# data + preprocessing Auto= pipeline
from sklearn.pipeline import Pipeline, make_pipeline

import warnings

warnings.filterwarnings('ignore')

# data
dataset = load_diabetes()

x = dataset.data
y = dataset.target
print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# model
# 파이프라인 입력은 스탭스= 리스트, 튜플, 키워드 형식으로 작성한다.
# model = Pipeline([("Minmax", MinMaxScaler()), ('model', SVC())])

# 파이프라인 입력은 스탭스= 리스트 형식으로 작성한다.
models = [LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]
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
전처리 :  MinMaxScaler() 모델 : LinearRegression()
results : 0.4384360401733268
전처리 :  StandardScaler() 모델 : LinearRegression()
results : 0.4384360401733268
전처리 :  MinMaxScaler() 모델 : KNeighborsRegressor()
results : 0.2616686952956607
전처리 :  StandardScaler() 모델 : KNeighborsRegressor()
results : 0.2602895052339007
전처리 :  MinMaxScaler() 모델 : DecisionTreeRegressor()
results : -0.14181891097282429
전처리 :  StandardScaler() 모델 : DecisionTreeRegressor()
results : -0.26593378742593066
전처리 :  MinMaxScaler() 모델 : RandomForestRegressor()
results : 0.29802580574097026
전처리 :  StandardScaler() 모델 : RandomForestRegressor()
results : 0.3090488878502915
"""
