#  feature_importances_
#  max_depth

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

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

dataset = load_boston()


x = dataset.data
y = dataset.target
df = pd.DataFrame(x, columns=dataset.feature_names)
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(df.columns)
"""

"""
# 필요없는 컬럼 제거
del df["INDUS"]
del df["RAD"]
del df["ZN"]
del df["CHAS"]


# 스플릿 하기전에 넘파이로
x = df.to_numpy()

# 1 data
x_train, x_test, y_train, y_test = train_test_split(
    x, dataset.target, train_size=0.8, random_state=44
)
print(x_train.shape)
# 2 model
model = GradientBoostingRegressor(max_depth=8)

# 3 fit
model.fit(x_train, y_train)

# 4 evel
score = model.score(x_test, y_test)

print(model.feature_importances_)
print("score :", score)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    # plt 프래임 크기 설정
    plt.figure(figsize=(10,6))
    print(x.data.shape[1]) # 8
    # y ticks 길이는 x 컬럼의 길이
    n_features = x.data.shape[1]
    # x 컬럼의 길이만큼 바그래프 생성 벨류는  model.feature_importances 값을 입력
    # 위치는 센터
    plt.barh(np.arange(n_features), model.feature_importances_,
        align='center')
    # yticks 의 길이는 x 컬럼의 길이 이름은 df.columns 리스트
    plt.yticks(np.arange(n_features), df.columns)
    # x 라벨
    plt.xlabel("Feature Importances")
    # y 라벨
    plt.ylabel("Features")
    # y 축 길이는 -1 ~ x 컬럼 수 만큼 가변
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

print(cut_columns(model.feature_importances_, df.columns, 4 ))


"""
[3.98558882e-02 5.84638400e-04 5.22788418e-03 1.10649256e-04
 3.07121879e-02 5.44853512e-01 1.13191924e-02 8.98143493e-02
 1.75605899e-03 1.31284140e-02 2.50815667e-02 1.59610050e-02
 2.21594654e-01]
acc : 0.8749166240071206

[0.0377973  0.03482545 0.54642647 0.01200885 0.09033774 0.01408746
 0.02660438 0.0179329  0.21997944]
acc : 0.8718807888696043
"""