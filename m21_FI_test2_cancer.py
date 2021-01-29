#  feature_importances_
#  max_depth

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()


x = dataset.data
y = dataset.target
df = pd.DataFrame(x, columns=dataset.feature_names)
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(df.columns)
"""

"""
# 필요없는 컬럼 제거
del df["worst fractal dimension"]
del df["worst symmetry"]
del df["worst concavity"]
del df["worst compactness"]
del df["worst area"]
del df["fractal dimension error"]
del df["symmetry error"]
del df["concave points error"]
del df["compactness error"]
del df["smoothness error"]
del df["perimeter error"]
del df["texture error"]
del df["radius error"]
del df["mean fractal dimension"]
del df["mean symmetry"]
del df["mean concave points"]
del df["mean concavity"]
del df["mean compactness"]
del df["mean smoothness"]
del df["mean perimeter"]
del df["mean texture"]
del df["mean radius"]

# 스플릿 하기전에 넘파이로
x = df.to_numpy()

# 1 data
x_train, x_test, y_train, y_test = train_test_split(
    x, dataset.target, train_size=0.8, random_state=44
)
print(x_train.shape)
# 2 model
model = DecisionTreeClassifier(max_depth=8)

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

print(df.columns)
"""
Index(['mean area', 'area error', 'concavity error', 'worst radius',
       'worst texture', 'worst perimeter', 'worst smoothness',
       'worst concave points'],
      dtype='object')
"""

"""
[0.         0.         0.         0.         0.         0.
 0.00621763 0.         0.         0.         0.         0.
 0.         0.00299321 0.         0.0186973  0.         0.00621763
 0.00621763 0.         0.         0.05204668 0.715762   0.00925886
 0.04409158 0.         0.         0.13849747 0.         0.        ]
acc : 0.9385964912280702

[0.         0.01542847 0.         0.01547649 0.06079023 0.71820465
 0.0484387  0.14166147]
acc : 0.9385964912280702
"""