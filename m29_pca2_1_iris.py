import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)         # (442, 10) (442,)

# # 컬럼을 n_components 수 만큼 압축한다
# pca = PCA(n_components=1)
# x2 = pca.fit_transform(x)
# print(x2.shape)                 # (442, 7)

# # 컬럼을 압축한 컬럼의 변화 비율을 확인
# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))

"""
print(sum(pca_EVR))

n_components = 7
0.9479436357350414
n_components = 8
0.9913119559917797
n_components = 9
0.9991439470098977
"""

pca = PCA(n_components=3)
pca.fit(x)
# cumsum = 누적합계
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
d = np.argmax(cumsum >= 0.95)+1
# print(np.argmax(cumsum >= 0.95))
print("cumsum >= 0.95", cumsum >= 0.95)
print("d :", d)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

x = pca.fit_transform(x)

# 1 data
x_train, x_test, y_train, y_test = train_test_split(
    x, dataset.target, train_size=0.8, random_state=44
)
# 2 model
model = XGBClassifier(n_jobs=-1,use_label_encoder=False)

# 3 fit
model.fit(x_train, y_train,eval_metric='logloss')

# 4 evel
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc :", acc)
"""
n_components : 0
acc : 0.9666666666666667
n_components : 3
acc : 0.9333333333333333
"""