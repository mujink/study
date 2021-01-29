#  feature_importances_
#  max_depth

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
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

dataset = load_breast_cancer()


x = dataset.data
y = dataset.target
df = pd.DataFrame(x, columns=dataset.feature_names)
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(df.columns)
"""

"""
# 필요없는 컬럼 제거

del df["symmetry error"]
del df["concave points error"]
del df["smoothness error"]
del df["texture error"]
del df["concavity error"]
del df["mean fractal dimension"]
del df["mean symmetry"]
del df["mean smoothness"]


# 스플릿 하기전에 넘파이로
x = df.to_numpy()

# 1 data
x_train, x_test, y_train, y_test = train_test_split(
    x, dataset.target, train_size=0.8, random_state=44
)
print(x_train.shape)
# 2 model
model = RandomForestClassifier(max_depth=8)

# 3 fit
model.fit(x_train, y_train)

# 4 evel
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc :", acc)

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

print(cut_columns(model.feature_importances_, df.columns,8))

"""
[0.04307756 0.01292705 0.06472321 0.03118067 0.00908127 0.00916189
 0.05715332 0.14704794 0.00500985 0.00308658 0.00677494 0.0033952
 0.01041879 0.02739923 0.00830583 0.00534702 0.00600295 0.0037944
 0.00347454 0.00476408 0.0830188  0.01844267 0.17520663 0.07894903
 0.01851912 0.01047282 0.0347422  0.10366345 0.00783163 0.00702735]
acc : 0.9649122807017544

[0.03189444 0.01865945 0.06950647 0.05539338 0.00763593 0.06200716
 0.15100585 0.00689237 0.00784923 0.02003768 0.00767442 0.00776196
 0.09900078 0.02302197 0.11857166 0.14260764 0.01719985 0.02850067
 0.01956918 0.08130944 0.01147028 0.01243018]
acc : 0.9649122807017544
"""