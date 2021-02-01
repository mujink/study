# 데이터 별로 5 개 만든다.

#  feature_importances_
#  max_depth

import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()


x = dataset.data
y = dataset.target

# 1 data
kfold = KFold(n_splits=5, shuffle=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, dataset.target, train_size=0.8, random_state=44
)
print(x_train.shape)

parameters = [
    {"n_estimators" : [100,200,300], "learning_rate" : [0.1,0.3,0.001,0.01],
    "max_depth":[4,5,6]},
    {"n_estimators" : [90,100,110], "learning_rate" : [0.1,0.001,0.01],
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators" : [90,110], "learning_rate" : [0.1,0.001,0.5],
    "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1],
    "colsample_bylevel":[0.6,0.7,0.9]},
]
# n_estimators : 반복 학습하는 횟수, 생성할 트리의 갯수, 값만큼 부스팅 진행하되 xgb의 경우 로스 개선이 없으면 부스팅 수행을 중지함
# learning_rate : 경사하강시 길이, 아담, 나담등 엑티베이션 함수의 학습율로 들어감
# max_depth : 의사결정 트리 기반 알고리즘 트리의 깊이, 레이어를 의미함  0을 지정하면  깊이의 제한이 없음.
# colsample_bytree : 트리 생성시 훈련데이터의 특성을 샘플링하는 비율, 피처가 많을 때 과적합 조절에 사용
# colsample_bylevel : 트리의 레벨별 훈련 데이터 변수를 샘플링하는 비율

# eval_metric 의 종류
"""
eval_metric : val의 평가지표
rmse : 제곱 평균 제곱근 오차
mae : 절대 오류를 의미
logloss : 확률분포의 음의로그 값
error : 이진 분류 오류율로 acc의 반대로 개념임 (0.5 임계 값)
merror : 다중 클래스 분류에 대한 error임
mlogloss : 다중 클래스 logloss
auc : Area under the curve error와 acc중 acc가 차지하는 비율
""" 

import datetime
start = datetime.datetime.now()

# 2 model
for train_index, test_index in kfold.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = GridSearchCV(XGBClassifier(n_jobs=-1,use_label_encoder=False), parameters, cv=kfold)
# 3 fit
    model.fit(x_train, y_train,eval_metric='logloss')
# 4 evel
    acc = model.score(x_test, y_test)
    print("acc :", acc)

end = datetime.datetime.now()
print(end - start)    

import matplotlib.pyplot as plt
import numpy as np

# def plot_feature_importances_dataset(model):
#     # plt 프래임 크기 설정
#     plt.figure(figsize=(10,6))
#     print(x.data.shape[1]) # 8
#     # y ticks 길이는 x 컬럼의 길이
#     n_features = x.data.shape[1]
#     # x 컬럼의 길이만큼 바그래프 생성 벨류는  model.feature_importances 값을 입력
#     # 위치는 센터
#     plt.barh(np.arange(n_features), model.feature_importances_,
#         align='center')
#     # yticks 의 길이는 x 컬럼의 길이 이름은 df.columns 리스트
#     plt.yticks(np.arange(n_features), df.columns)
#     # x 라벨
#     plt.xlabel("Feature Importances")
#     # y 라벨
#     plt.ylabel("Features")
#     # y 축 길이는 -1 ~ x 컬럼 수 만큼 가변
#     plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model)

# plot_importance(model)
# plt.show()



"""
[0.00581062 0.01234676 0.68648463 0.29535798]
acc : 0.9666666666666667

[0.0160174  0.61124271 0.37273989]
acc : 0.9666666666666667

xgbc
[0.02323038 0.01225644 0.8361378  0.12837538]
acc : 0.9666666666666667

n_jobs = [-1,8,4,1]
0:00:00.104709
0:00:00.036901
0:00:00.057845
0:00:00.039893

GridSearchCV
acc : 0.9
acc : 0.9666666666666667
acc : 0.9333333333333333
acc : 0.9666666666666667
acc : 0.9666666666666667
0:05:37.436659

"""