#  feature_importances_
#  max_depth

import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

dataset = load_breast_cancer()


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

    model = XGBClassifier(n_jobs=n, use_label_encoder=False)

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
[1.11882680e-06 1.27653893e-02 2.89361344e-03 5.59012723e-04
 1.84196435e-05 4.11339126e-03 8.31508689e-04 4.07408539e-01
 1.21303933e-03 2.72747914e-04 4.23321009e-03 1.75746559e-03
 1.57739853e-03 7.57530073e-03 3.72306479e-04 5.46570022e-04
 1.85398002e-03 1.07821108e-03 4.07121260e-04 3.46114983e-03
 6.97278658e-02 5.80282347e-02 2.89120196e-01 4.10145897e-02
 1.39304261e-02 5.54466382e-04 2.04151492e-02 5.37616402e-02
 2.18592729e-04 2.89345335e-04]
acc : 0.9736842105263158

[7.83106012e-03 4.31121618e-04 1.12875936e-03 4.43547451e-03
 1.14725955e-03 4.08675270e-01 3.19341770e-03 4.41100620e-03
 3.21511186e-03 1.25347880e-03 7.61730808e-03 3.04727387e-04
 1.07501721e-03 1.94621463e-03 7.15116128e-02 6.30090791e-02
 2.80415393e-01 5.06517417e-02 1.53955278e-02 3.12172617e-04
 1.93544647e-02 5.26847811e-02]
acc : 0.9649122807017544

xbgc
[4.9588699e-03 1.5423428e-02 6.3683779e-04 2.2023508e-02 7.9919444e-03
 9.5059077e-04 1.1493003e-02 1.5286781e-01 3.9727203e-04 2.7880725e-03
 3.4555981e-03 5.6501115e-03 2.8066258e-03 3.8451280e-03 2.8832396e-03
 3.3951809e-03 3.4120989e-03 5.8727746e-04 9.2994876e-04 3.7213673e-03
 1.1663227e-01 2.0733794e-02 5.1057857e-01 1.3440680e-02 6.8047098e-03
 0.0000000e+00 2.1567145e-02 5.5507991e-02 1.3584920e-03 3.1585069e-03]
acc : 0.9824561403508771

n_jobs = [-1,8,4,1]
0:00:00.076057
0:00:00.031040
0:00:00.034078
0:00:00.070838
"""