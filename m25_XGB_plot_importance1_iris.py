#  feature_importances_
#  max_depth

import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance

dataset = load_iris()


x = dataset.data
y = dataset.target

# 1 data
x_train, x_test, y_train, y_test = train_test_split(
    x, dataset.target, train_size=0.8, random_state=44
)

n_jobs = [-1,8,4,1]
for n in n_jobs:

    import datetime
    start = datetime.datetime.now()


# 2 model

    model = XGBClassifier(n_jobs=8, use_label_encoder=False)

# 3 fit
    model.fit(x_train, y_train,eval_metric='logloss')

# 4 evel
    acc = model.score(x_test, y_test)

    # print(model.feature_importances_)
    # print("acc :", acc)
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

plot_importance(model)
plt.show()

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
"""