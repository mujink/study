import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from xgboost import XGBClassifier

# 불러오기
train = pd.read_csv('../data/csv/Dacon2/data/train.csv',header = 0)
test = pd.read_csv('../data/csv/Dacon2/data/test.csv',header = 0)
# ==============================그림보기==========================
# 인덱스
idx = 318
# 트레인 데이터의 인덱스 길이만큼의 데이터에 대해 0번 컬럼의 값을 28,28로 쉐이프하여 이미지로 초기화
img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
# 트레인 데이터의 인덱스 길이만큼 타겟을 디지트로 초기화
digit = train.loc[idx, 'digit']
# 트레인 데이터의 인덱스 길이만큼 레터를 레터로 초기화
letter = train.loc[idx, 'letter']

# 타이틀을 인덱스, 디지트, 레터로 함
plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
# 인덱스 번째의 쉐이프한 이미지를 이미지로 출력
plt.imshow(img)
# 보기
plt.show()
# ==============================데이터 전처리==========================

# 트래인의 아이디, 디지트, 레터 컬럼을 제거한 나머지 컬럼의 값을 축변경하여 초기화
x_train = train.drop(['id', 'digit', 'letter'], axis=1).values

# from sklearn.decomposition import PCA

# pca = PCA()
# x2 = pca.fit_transform(x_train)
# print(x2.shape)                 # (70000, 784)

# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("누계 :", cumsum)
# d = np.argmax(cumsum >= 0.99)+1
# print("cumsum >= 0.95", cumsum >= 0.95)
# print("d :", d)


y_train = train['digit'].values




from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)


#  모델을 정의하고 변수 트래인을 인자로 받음
# parameters = [
#     {"n_estimators" : [100,200,300], "learning_rate" : [0.1,0.3,0.001,0.01],
#     "max_depth":[4,5,6]},
#     {"n_estimators" : [90,100,110], "learning_rate" : [0.1,0.001,0.01],
#     "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
#     {"n_estimators" : [90,110], "learning_rate" : [0.1,0.001,0.5],
#     "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1],
#     "colsample_bylevel":[0.6,0.7,0.9]},
# ]

# parameters = [
#     {"XGB__n_estimators" : [100,200,300], "XGB__learning_rate" : [0.1,0.3,0.001,0.01],
#     "XGB__max_depth":[4,5,6]},
#     {"XGB__n_estimators" : [90,100,110], "XGB__learning_rate" : [0.1,0.001,0.01],
#     "XGB__max_depth":[4,5,6], "XGB__colsample_bytree":[0.6,0.9,1]},
#     {"XGB__n_estimators" : [90,110], "XGB__learning_rate" : [0.1,0.001,0.5],
#     "XGB__max_depth":[4,5,6], "XGB__colsample_bytree":[0.6,0.9,1],
#     "XGB__colsample_bylevel":[0.6,0.7,0.9]},
# ]

x_test = test.drop(['id', 'letter'], axis=1).values
print(x_test.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
scaler.fit(x_train)
x = scaler.transform(x_train)
x = scaler.transform(x_test)
x = scaler.transform(x_val)

from sklearn.model_selection import RandomizedSearchCV

# model = RandomizedSearchCV(XGBClassifier(n_jobs=8,use_label_encoder=False), parameters)

model = XGBClassifier(n_estimators=3000, learning_rate=0.001, n_jobs=8,max_depth= 10, colsample_bytree=1,colsample_bylevel=1)

model.fit(x_train, y_train, verbose=1, eval_metric=['mlogloss','merror','cox-nloglik'],
        eval_set=[(x_train,y_train),(x_val,y_val)]
        # early_stopping_rounds=100
            )

r2 = model.score(x_val,y_val)
print("r2 :", r2)


result = model.evals_result()
# 발리데이션 로스 값을 출력

import matplotlib.pyplot as plt

epochs = len(result['validation_0']['cox-nloglik'])
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['cox-nloglik'], label='Train')
ax.plot(x_axis, result['validation_1']['cox-nloglik'], label='Test')
ax.legend()
plt.ylabel('cox-nloglik')
plt.title('XGBoost cox-nloglik')

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['merror'], label='Train')
ax.plot(x_axis, result['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('merror')
plt.title('XGBoost merror')

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, result['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('mlogloss')
plt.title('XGBoost mlogloss')

plt.show()
# ============================================


submission = pd.read_csv('../data/csv/Dacon2/data/submission.csv')
submission['digit'] = np.argmax(model.predict(x_test))
print(submission.head())

# submission.to_csv('../data/csv/Dacon2/data/baseline.csv', index=False)
