#  feature_importances_
#  max_depth

import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def cut_columns(feature_importances,columns,number):
    temp = []
    for i in feature_importances:
        temp.append(i)
    temp.sort()
    temp=temp[:number]
    result = []
    for j in temp:
        index = feature_importances.tolist().index(j)
        result.append(columns[index])
    return result
# print(cut_columns(model.feature_importances_, df.columns, 4 ))

dataset = load_diabetes()


x = dataset.data
y = dataset.target
df = pd.DataFrame(x, columns=dataset.feature_names)
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(df.columns)

# 필요없는 컬럼 제거
# del df["sex"]
# del df["s4"]
# del df["s3"]

# 스플릿 하기전에 넘파이로
x = df.to_numpy()

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

# print(cut_columns(model.feature_importances_, df.columns, 4 ))


"""
[0.08490457 0.01029597 0.23429762 0.05704266 0.05301435 0.07205492
 0.03992525 0.02114709 0.36642795 0.06088962]
score : 0.16861223874427433

[0.08873816 0.24253146 0.07155385 0.06785732 0.08333687 0.37099206
 0.07499027]
score : 0.16857157651081633

xgbr
[0.0368821  0.03527097 0.15251055 0.05477958 0.04415327 0.06812558
 0.0651588  0.05049536 0.42164674 0.0709771 ]
acc : 0.24138193114785134

0:00:00.097958
0:00:00.056847
0:00:00.053879
0:00:00.077791
"""