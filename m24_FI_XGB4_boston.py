

import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

dataset = load_boston()


x = dataset.data
y = dataset.target


# 1 data
x_train, x_test, y_train, y_test = train_test_split(
    x, dataset.target, train_size=0.8, random_state=44
)


n_jobs = [-1,8,4,1]
for n in n_jobs:
    # print(x_train.shape)

    import datetime
    start = datetime.datetime.now()


# 2 model
# model = GradientBoostingClassifier()

    model = XGBRegressor(n_jobs=n, use_label_encoder=False)

# 3 fit
    model.fit(x_train, y_train, eval_metric='logloss')

    # 4 evel
    # acc = model.score(x_test, y_test)

    # print(model.feature_importances_)
    # print("acc :", acc)
    end = datetime.datetime.now()

    print(end - start)

# import matplotlib.pyplot as plt
# import numpy as np

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
# plt.show()



"""
[3.98558882e-02 5.84638400e-04 5.22788418e-03 1.10649256e-04
 3.07121879e-02 5.44853512e-01 1.13191924e-02 8.98143493e-02
 1.75605899e-03 1.31284140e-02 2.50815667e-02 1.59610050e-02
 2.21594654e-01]
acc : 0.8749166240071206

[0.0377973  0.03482545 0.54642647 0.01200885 0.09033774 0.01408746
 0.02660438 0.0179329  0.21997944]
acc : 0.8718807888696043

xgbr
[0.01311134 0.00178977 0.00865051 0.00337766 0.03526587 0.24189197
 0.00975884 0.06960727 0.01454236 0.03254252 0.04658296 0.00757505
 0.51530385]
acc : 0.8902902185916939

0:00:00.096740
0:00:00.056541
0:00:00.058842
0:00:00.082705

"""