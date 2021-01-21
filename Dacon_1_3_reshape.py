# TrainDbSet 불러오기
# x, y = target1, y = target2 초기화하기
# x, y => train, test, val스플릿 하기

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# TrainDbSet 불러오기 ========================================================================================
fitset = pd.read_csv('../data/csv/Dacon/preprocess_csv/TrainDbSet2.csv', encoding='ms949', index_col=0)
# train shape (52464, 9)

# 상관계수 확인===============================================================================================
print(fitset.corr())
sns.set(font_scale=0.7)
sns.heatmap(data=fitset.corr(), square=True, annot=True, cbar=True)
plt.show()

# 컬럼 선택 하기 =================================================================================================


seq = np.array(fitset)
x = seq[:,0:-2]
y = seq[:,-2:]

print(x.shape)
print(y.shape)

# # #  넘파이 저장=======================================================
print(x.shape)
print(y.shape)
np.save('../data/csv/Dacon/np/TrainDb_X.npy',x)
np.save('../data/csv/Dacon/np/TrainDb_Y.npy',y)

# # #  넘파이 불러오기=======================================================

# x = np.load('../data/csv/Dacon/np/TrainDb_X.npy',allow_pickle=True)
# y = np.load('../data/csv/Dacon/np/TrainDb_Y.npy',allow_pickle=True)

