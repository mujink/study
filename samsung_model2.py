#  삼성 모델1에서 모델의 아웃풋 레이어를 2개로 만들었음.

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


y1_train = np.array(y_train[:,0])
y2_train = np.array(y_train[:,1])
y1_test = np.array(y_test[:,0])
y2_test = np.array(y_test[:,1])
y1_val =  np.array(y_val[:,0])
y2_val =  np.array(y_val[:,1])
# 모델===============================================================================

# #2.1 sam_model1
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

#2.4 model1 Branch of Output1
output1 = Dense(30)(dense3)
output1 = Dense(20)(output1)
output1 = Dense(10)(output1)
output1 = Dense(1)(output1)

#2.5 model2 Branch of Output2
output2 = Dense(30)(dense3)
output2 = Dense(20)(output2)
output2 = Dense(10)(output2)
output2 = Dense(1)(output2)


# 2.6 def Model1,2
model = Model(inputs=[input1, input2],outputs=[output1,output2])
model.summary()

# 콜백 ===============================================================================
modelpath = "./hdf5/sam_cdc_ens.hdf5"
es = EarlyStopping(monitor = 'val_loss',patience=100, mode="min")
cp = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath, save_best_only=True, mode='min')

model = load_model('./hdf5/sam_cdc_ens.hdf5')
# model.compile(loss = 'mse',optimizer = 'adam')
hist = model.fit(x=[x1_train,x2_train],y=[y1_train,y2_train],  epochs= 10000, batch_size=64, verbose=1, validation_data=([x1_val,x2_val], [y1_val,y2_val]),callbacks=[es,cp], shuffle=True)

# # 로스 시각화
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
loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test], batch_size=64)
print("loss : ",loss)

y1_predict, y2_predict = model.predict([x1_test, x2_test])

r2_1 = r2_score(y1_test, y1_predict)
print("R2_1 : ", r2_1)

r2_2 = r2_score(y2_test, y2_predict)
print("R2_1 : ", r2_2)

D_1 = model.predict([x1_prd,x2_prd])
print("=====예상 값=====")
print("D_1 : ",D_1)

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
모델 1 보다 더 잘나옴.
loss :  [10692631.0, 5435408.5, 5257222.0]
R2_1 :  0.9247115174403506
R2_1 :  0.9255013804914275
=====예상 값=====
D_1 :  [array([[94361.19]], dtype=float32), array([[94806.15]], dtype=float32)]

loss :  [7135490.5, 3448997.75, 3686493.75]
R2_1 :  0.9522262525281997
R2_1 :  0.9477597288266737
=====예상 값=====
D_1 :  [array([[94054.336]], dtype=float32), array([[94108.766]], dtype=float32)]

loss :  [16202129.0, 7869245.5, 8332883.5]
R2_1 :  0.8909992842762209
R2_1 :  0.8819170123265946
=====예상 값=====
D_1 :  [array([[90425.47]], dtype=float32), array([[90430.98]], dtype=float32)]
"""