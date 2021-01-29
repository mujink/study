#  feature_importances_
#  max_depth

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

dataset = load_boston()


x = dataset.data
y = dataset.target
df = pd.DataFrame(x, columns=dataset.feature_names)
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(df.columns)
"""

"""
# 필요없는 컬럼 제거

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
model = RandomForestRegressor(max_depth=8)

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

"""

"""
[0.02017336 0.00362587 0.00233176 0.00556109 0.00090131 0.0073343
 0.46817136 0.0009569  0.00061335 0.05948964 0.03313522 0.16226654
 0.23543932]
acc : 0.8864803394625177

[0.03917877 0.00583637 0.02098039 0.40616683 0.01157567 0.07847024
 0.00397491 0.01295787 0.01505912 0.01014301 0.39565683]
acc : 0.8862787434114875
"""