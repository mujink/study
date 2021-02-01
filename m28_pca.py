import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)         # (442, 10) (442,)

# 컬럼을 n_components 수 만큼 압축한다
pca = PCA(n_components=9)
x2 = pca.fit_transform(x)
print(x2.shape)                 # (442, 7)

# 컬럼을 압축한 컬럼의 변화 비율을 확인
pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))

"""
n_components = 7
0.9479436357350414
n_components = 8
0.9913119559917797
n_components = 9
0.9991439470098977
"""