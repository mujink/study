# TrainDbSet 불러오기
# x, y = target1, y = target2 초기화하기
# x, y => train, test, val스플릿 하기

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# TrainDbSet 불러오기 ========================================================================================
fitset = pd.read_csv('../data/csv/Dacon/preprocess_csv/TrainDbSet2.csv', encoding='ms949', index_col=0)
prdset = pd.read_csv('../data/csv/Dacon/preprocess_csv/TestDbSet2.csv', encoding='ms949', index_col=0)

# train shape (52464, 9)

# 상관계수 확인===============================================================================================
print(fitset.corr())
sns.set(font_scale=0.7)
sns.heatmap(data=fitset.corr(), square=True, annot=True, cbar=True)
plt.show()

# 컬럼 선택 하기 =================================================================================================
fitset = fitset.to_numpy()
prdset = prdset.to_numpy()

def split_xy(data,timestep):
    x, y1, y2 = [],[],[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end,:-2]
        tmp_y1 = data[x_end-1:x_end,-2]
        tmp_y2 = data[x_end-1:x_end,-1]
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return(np.array(x),np.array(y1),np.array(y2))
    return(np.array(x),np.array(y1))

x,y1,y2  = split_xy(fitset,1)

def split_x(data,timestep):
    x = []
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end]
        x.append(tmp_x)
    return(np.array(x))

xt = split_x(prdset,1)
# xt = split_x(prdset,1)

# x = x.reshape(-1,7,1)
# y1 = y1.reshape(-1,1)
# y2 = y2.reshape(-1,1)
# xt = xt.reshape(-1,7,1)

# x = x.reshape(-1,24,2,10).astype('float32')
# y1 = y1.reshape(-1,48,2).astype('float32')
# y2 = y2.reshape(-1,48,2).astype('float32')

print(x.shape)
print(y1.shape)
print(y2.shape)
print(xt.shape)
# x = np.transpose(x)
# y1 = np.transpose(y1)
# y2 = np.transpose(y2)
# print(x.shape)
# print(y1.shape)
# print(y2.shape)
# # #  넘파이 저장=======================================================


np.save('../data/csv/Dacon/np/TrainDb_X.npy',x)
np.save('../data/csv/Dacon/np/TrainDb_Y1.npy',y1)
np.save('../data/csv/Dacon/np/TrainDb_Y2.npy',y2)
np.save('../data/csv/Dacon/np/prdDb_Xt.npy',xt)

# # #  넘파이 불러오기=======================================================

# x = np.load('../data/csv/Dacon/np/TrainDb_X.npy',allow_pickle=True)
# y = np.load('../data/csv/Dacon/np/TrainDb_Y.npy',allow_pickle=True)
# xt = np.load(../data/csv/Dacon/np/prdDb_Xt.npy'',allow_pickle=True)

# # # 샘플 보기=======================================================

# x = pd.DataFrame(x[1])
# y = pd.DataFrame(y[1])
# xt = pd.DataFrame(xt[1])
# x.to_csv('../data/csv/Dacon/preprocess_csv/XDbSet2.csv', encoding='ms949', sep=",")
# y.to_csv('../data/csv/Dacon/preprocess_csv/YDbSet2.csv', encoding='ms949', sep=",")
# xt.to_csv('../data/csv/Dacon/preprocess_csv/XpDbSet2.csv', encoding='ms949', sep=",")

