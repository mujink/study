#  feature_importances_
#  max_depth

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
del df["smoothness error"]
del df["compactness error"]
del df["mean fractal dimension"]
del df["worst fractal dimension"]
del df["worst symmetry"]
del df["mean smoothness"]
del df["mean radius"]


# 스플릿 하기전에 넘파이로
x = df.to_numpy()

# 1 data
x_train, x_test, y_train, y_test = train_test_split(
    x, dataset.target, train_size=0.8, random_state=44
)
print(x_train.shape)
# 2 model
model = GradientBoostingClassifier()

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
"""