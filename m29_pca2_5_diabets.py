import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

dataset = load_diabetes()
x = dataset.data
y = dataset.target

"""

# 컬럼을 n_components 수 만큼 압축한다
pca = PCA(n_components=1)
x2 = pca.fit_transform(x)
print(x2.shape)                 # (442, 7)

# 컬럼을 압축한 컬럼의 변화 비율을 확인
pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))


n_components = 7
0.9479436357350414
n_components = 8
0.9913119559917797
n_components = 9
0.9991439470098977
"""

pca = PCA(n_components=6)
# pca = PCA()
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

"""

"""

# 1 data
x_train, x_test, y_train, y_test = train_test_split(
    x, dataset.target, train_size=0.8, random_state=44
)
print(x_train.shape)
# 2 model
model = XGBRegressor(n_jobs=-1,use_label_encoder=False)

# 3 fit
model.fit(x_train, y_train,eval_metric='logloss')

# 4 evel
score = model.score(x_test, y_test)

print(model.feature_importances_)
print("score :", score)
"""
n_components : 0
score : 0.3044465654517613
n_components : 6
score : 0.23195281590722838
"""