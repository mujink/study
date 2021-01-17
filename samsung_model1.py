

from typing import ValuesView
import numpy as np
from numpy.core.numeric import False_
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense,LSTM,Conv1D,MaxPool1D,Dropout,Flatten, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# #  넘파이 불러오기=======================================================
# X1 : (1079, 5, 6)
# X2 : (1079, 5, 5)
# Y: (1079, 2)

# X1 Train : (690, 5, 6) X2 Trian : (690, 5, 5)
# Y1 Train : (690, 2) <class 'numpy.ndarray'>
# X1 Tset  : (216, 5, 6)  X2 Test  : (216, 5, 5)
# Y1 Tset  : (216, 2) <class 'numpy.ndarray'>
# X1 Val   : (173, 5, 6)   X2 val   : (173, 5, 5)
# Y1 Val   : (173, 2) <class 'numpy.ndarray'>
# X1_prd   : (1, 5, 6)   X2_prd   : (1, 5, 5)

x1_train = np.load('./npy/1.npy',allow_pickle=True)
x2_train = np.load('./npy/2.npy',allow_pickle=True)
y_train = np.load('./npy/3.npy',allow_pickle=True)
x1_test = np.load('./npy/11.npy',allow_pickle=True)
x2_test = np.load('./npy/12.npy',allow_pickle=True)
y_test = np.load('./npy/13.npy',allow_pickle=True)
x1_val = np.load('./npy/21.npy',allow_pickle=True)
x2_val = np.load('./npy/22.npy',allow_pickle=True)
y_val = np.load('./npy/23.npy',allow_pickle=True)
x1_prd = np.load('./npy/31.npy',allow_pickle=True)
x2_prd = np.load('./npy/32.npy',allow_pickle=True)



# 모델===============================================================================

#2.1 sam_model1
input1 = Input(shape=(x1_train.shape[1],x1_train.shape[2]))
LSTM1 = LSTM(128, activation='relu', return_sequences=True)(input1)
LSTM1 = LSTM(128, activation='relu')(LSTM1)
dense1 = Dense(128,activation='relu')(LSTM1)
dense1 = Dense(128,activation='relu')(dense1)
dense1 = Dense(128,activation='relu')(dense1)
dense1 = Dense(128,activation='relu')(dense1)

#2.2 cdc_model2
input2 = Input(shape=(x2_train.shape[1],x2_train.shape[2]))
LSTM2 = LSTM(128, activation='relu', return_sequences=True)(input2)
LSTM2 = LSTM(128, activation='relu')(LSTM2)
dense2 = Dense(128,activation='relu')(LSTM1)
dense2 = Dense(128,activation='relu')(dense2)
dense2 = Dense(128,activation='relu')(dense2)
dense2 = Dense(128,activation='relu')(dense2)

#2.3 model Concatenate
from tensorflow.keras.layers import concatenate
merge1 = concatenate([dense1, dense2])
dense3 = Dense(128,activation='relu')(merge1)
dense3 = Dense(128,activation='relu')(dense3)
dense3 = Dense(128,activation='relu')(dense3)
dense3 = Dense(64,activation='relu')(dense3)
dense3 = Dense(32,activation='relu')(dense3)
dense3 = Dense(16,activation='relu')(dense3)
dense3 = Dense(8,activation='relu')(dense3)
dense3 = Dense(4,activation='relu')(dense3)
dense3 = Dense(2,activation='relu')(dense3)
output = Dense(2,activation='relu')(dense3)


# # # #2.6 def Model1,2
model = Model(inputs=[input1, input2],outputs=output)
model.summary()

# # 콜백 ===============================================================================
modelpath = "./hdf5/sam_cdc_ens.hdf5"
es = EarlyStopping(monitor = 'val_loss',patience=100, mode="min")
cp = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath, save_best_only=True, mode='min')

model = load_model('./hdf5/sam_cdc_ens.hdf5')
# model.compile(loss = 'mse',optimizer = 'adam')
hist = model.fit([x1_train,x2_train],y_train,  epochs= 10000, batch_size=64, verbose=1, validation_data=([x1_val,x2_val], y_val),callbacks=[es,cp], shuffle=True)
# 로스 시각화
plt.rc('font', family='Malgun Gothic')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss ')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train_loss','val_loss'])
plt.show()

# # 훈련 모델 가중치 테스트==============================
# # model.save('./h5/sam_cdc_ens_maw.h5')
# # model = load_model('./h5/sam_cdc_ens_maw.h5')
# # model.save_weights('./h5/sam_cdc_ens_weight.h5')

# model = load_model('./hdf5/sam_cdc_ens.hdf5')
loss = model.evaluate([x1_test,x2_test],y_test, batch_size=64)
print("loss : ",loss)

y1_predict = model.predict([x1_test, x2_test])

r2_1 = r2_score(y_test, y1_predict)
print("R2_1 : ", r2_1)

D_1 = model.predict([x1_prd,x2_prd])
print("=====예상 값=====")
print("D_1 : ",D_1)

# # model = load_model('./hdf5/sam_cdc_ens.hdf5')
# loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test], batch_size= 256)
# print("loss, mse : ",loss )

# y_predict = model.predict([x1_test,x2_test])
# print(y_predict)
# print(y_predict.shape)
# print([y1_test,y2_test])
# print([y1_test,y2_test].shape)

# r2_D1 = r2_score(y1_test, y_predict[0])
# print("R2_D1 : ", r2_D1)

# r2_D2 = r2_score(y2_test, y_predict[1])
# print("R2_D2 : ", r2_D2)

# plt.figure(figsize=(9,3.7))
# plt.subplot(2,1,1)
# plt.plot(y1_predict, label='y_predict')
# plt.plot(y2_predict, label='y_predict')
# plt.legend()

# plt.subplot(2,1,2)
# plt.plot(y1_test, label='y_test')
# plt.plot(y2_test, label='y_test')
# plt.legend()
# plt.show()
# # csv만들기
# y_test = pd.DataFrame([y1_test,y2_test])
# y_test['Target'] = y_predict
# y_test.to_csv('../data/csv/y_test.csv', encoding='ms949', sep=",")


"""
loss :  14245118.0
(216, 2)
R2_1 :  0.8003004365836625
=====예상 값=====
D_1 :  [[102649.45  102741.484]]

loss :  6037003.5
(216, 2)
R2_1 :  0.915335411918567
=====예상 값=====
D_1 :  [[90466.305 90563.52 ]]

loss :  7243348.5
(216, 2)
R2_1 :  0.8984560662173566
=====예상 값=====
D_1 :  [[94567.07 94655.57]]

loss :  7128702.0
R2_1 :  0.9000355417057337
=====예상 값=====
D_1 :  [[95476.47 95654.82]]

loss :  7147830.0
R2_1 :  0.8997807740066877
=====예상 값=====
D_1 :  [[92848.59  93003.234]]
"""