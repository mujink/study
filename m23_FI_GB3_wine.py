#  feature_importances_
#  max_depth

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
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
# print(cut_columns(model.feature_importances_, df.columns, 4 ))

dataset = load_wine()


x = dataset.data
y = dataset.target
df = pd.DataFrame(x, columns=dataset.feature_names)
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(df.columns)
"""
Index(['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
       'proanthocyanins', 'color_intensity', 'hue',
       'od280/od315_of_diluted_wines', 'proline'],
      dtype='object')
"""
# 필요없는 컬럼 제거

del df["proanthocyanins"]
del df["nonflavanoid_phenols"]
del df["total_phenols"]
del df["alcalinity_of_ash"]



# 스플릿 하기전에 넘파이로
x = df.to_numpy()

# 1 data
x_train, x_test, y_train, y_test = train_test_split(
    x, dataset.target, train_size=0.8, random_state=44
)
print(x_train.shape)
# 2 model
model = GradientBoostingClassifier(max_depth=8)

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

print(cut_columns(model.feature_importances_, df.columns, 4 ))

"""
[5.11729230e-03 4.38311603e-02 1.63675577e-02 2.79886337e-04
 7.16175744e-03 3.37669838e-04 1.05509164e-01 2.76619356e-03
 1.68201073e-03 2.47477757e-01 3.43115445e-02 2.40744123e-01
 2.94413883e-01]
acc : 0.8888888888888888

[0.00649267 0.04243746 0.02291219 0.00616512 0.10406499 0.24954709
 0.03402652 0.23893011 0.29542384]
acc : 0.8888888888888888
"""