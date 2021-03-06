
import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense,LSTM,Conv1D,MaxPool1D,Dropout,Flatten, Input , Activation, Reshape
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


#  넘파이 불러오기=======================================================

x = np.load('../data/csv/Dacon/np/TrainDb_X.npy',allow_pickle=True)
y = np.load('../data/csv/Dacon/np/TrainDb_Y.npy',allow_pickle=True)

print(x.shape)
print(y.shape)
"""
(52464, 5)
(52464, 2)
"""
# x = x.reshape(-1,48,10).astype('float32')
# y = y.reshape(-1,48,2).astype('float32')
print(x.shape)
print(y.shape)
"""
(1093, 48, 5)
(1093, 48, 2)
"""
# train_test_split ==========================================================================================
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = False)#, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = False)#, random_state=1)
print("# shape  test=====================================================================================================")
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)

print(y_train.shape)
print(y_test.shape)
print(y_val.shape)
"""
(33576, 5)
(10493, 5)
(8395, 5)
(33576, 2)
(10493, 2)
(8395, 2)
"""
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Adam(lr=0.1)
# loss :  0.8494184613227844 y_pred :  [[12.966707]]

# optimizer = Adam(lr=0.01)
# loss :  0.009967544116079807 y_pred :  [[11.189517]]

# optimizer = Adam(lr=0.001)
# loss :  7.157154729986459e-11 y_pred :  [[11.000016]]

# optimizer = Adam(lr=0.0001)
# loss :  2.78911720670294e-05 y_pred :  [[10.997059]]
# ============================================================
# optimizer = Adadelta(lr=0.1)
# loss :  0.0032457425259053707 y_pred :  [[11.102851]]

# optimizer = Adadelta(lr=0.01)
# loss :  1.1453451406850945e-05 y_pred :  [[10.993094]]

# optimizer = Adadelta(lr=0.001)
# loss :  12.923612594604492 y_pred :  [[4.5785303]]
# ============================================================
# optimizer = Adamax(lr=0.1)
# loss :  32.24275207519531 y_pred :  [[15.299248]]

# optimizer = Adamax(lr=0.01)
# loss :  1.0700774034227978e-12 y_pred :  [[10.999999]]

# optimizer = Adamax(lr=0.001)
# loss :  1.3356395989205794e-08 y_pred :  [[10.999757]]
# ============================================================

# optimizer = Adagrad(lr=0.1)
# loss :  0.7076201438903809 y_pred :  [[11.1262045]]

optimizer = Adagrad(lr=0.01)
# loss :  9.321547622676007e-07 y_pred :  [[11.000604]]

# optimizer = Adagrad(lr=0.001)
# loss :  9.459159628022462e-06 y_pred :  [[10.995831]]

# optimizer = Adagrad(lr=0.001)
# loss :  2.1533718609134667e-05 y_pred :  [[10.995809]]
# ============================================================

# optimizer = RMSprop(lr=0.01)
# loss :  2.207491397857666 y_pred :  [[7.825623]]

# optimizer = RMSprop(lr=0.001)
# loss :  0.030038025230169296 y_pred :  [[10.646792]]

# optimizer = RMSprop(lr=0.0001)
# loss :  0.00043499600724317133 y_pred :  [[10.961597]]

# optimizer = RMSprop(lr=0.00001)
# loss :  5.387390046962537e-06 y_pred :  [[10.995292]]

# optimizer = RMSprop(lr=0.000001)
# loss :  6.924454689025879 y_pred :  [[6.2923384]]
# ============================================================

# optimizer = SGD(lr=0.01)
# loss :  nan y_pred :  [[nan]]

# optimizer = SGD(lr=0.001)
# loss :  7.841959813958965e-06 y_pred :  [[10.994494]]

# optimizer = SGD(lr=0.0001)
# loss :  0.0015226805116981268 y_pred :  [[10.952346]]
# ============================================================

optimizer = Nadam(lr=0.1)
# loss :  12.602835655212402 y_pred :  [[15.236937]]

# optimizer = Nadam(lr=0.01)
# loss :  9.379164112033322e-13 y_pred :  [[11.000004]] @@@@@@@@@@@@@@@@

# optimizer = Nadam(lr=0.001)
# loss :  2.149036591042597e-12 y_pred :  [[11.000001]] 

# optimizer = Nadam(lr=0.0001)
# loss :  5.9579133449005894e-06 y_pred :  [[10.995723]]
# ============================================================


quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 모델===============================================================================

# input1 = Input(shape=(x_train.shape[1],x_train.shape[2]))
input1 = Input(shape=(x_train.shape[1]))
# conv1d1 = Conv1D(64 ,padding='same')(input1)
# MxPl1d1 = MaxPool1D(pool_size=(2))(conv1d1)
# actve1 = Activation('relu')(MxPl1d1)
# conv1d1 = Conv1D(32,2 ,padding='same')(actve1)
# MxPl1d1 = MaxPool1D(pool_size=(2))(conv1d1)
# actve1 = Activation('relu')(MxPl1d1)
# Hlayer1 = LSTM(256, activation='relu',return_sequences=True)(input1)
# output1 = LSTM(256, activation='relu')(Hlayer1)
output1 = Dense(256, activation= 'relu')(input1)
output1 = Dense(128, activation= 'relu')(output1)
# output1 = Dense(128, activation= 'relu')(output1)
# output1 = Dense(48*2, activation= 'relu')(output1)
# output1 = Reshape((48,2))(output1)
output1 = Dense(64, activation= 'relu')(output1)
output1 = Dense(16, activation= 'relu')(output1)
output1 = Dense(8, activation= 'relu')(output1)
output1 = Dense(2, activation= 'relu')(output1)

# 2.6 def Model1,2
model = Model(inputs=input1, outputs=output1)
model.summary()

# # # 콜백 ===============================================================================
# modelpath = "../data/h5/Dacon_soler_cell.hdf5"
# es = EarlyStopping(monitor = 'val_loss',patience=100, mode="min")
# cp = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath, save_best_only=True, mode='min')
# reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=50, factor=0.5, verbose=1)
# model = load_model("../data/h5/Dacon_soler_cell.hdf5")
# # model.compile(loss = 'mse',optimizer = optimizer, metrics=['mae'])
# hist = model.fit (x_train, y_train,  epochs= 10000, batch_size=128, verbose=1, validation_data=(x_val , y_val),callbacks=[es,cp,reduce_lr], shuffle=False)

# # 로스 시각화
# plt.rc('font', family='Malgun Gothic')
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('loss ')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(['train_loss','val_loss'])
# plt.show()

# # 훈련 모델 가중치 테스트==============================
# # model.save('./h5/sam_cdc_ens_maw.h5')
# # model = load_model('./h5/sam_cdc_ens_maw.h5')
# # model.save_weights('./h5/sam_cdc_ens_weight.h5')

model = load_model('../data/h5/Dacon_soler_cell.hdf5')
loss = model.evaluate(x_test, y_test, batch_size=128)
print("loss : ",loss)

y_predict = model.predict(x_test)

print(y_test.shape)
print(y_predict.shape)
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)

print(y_predict)
print(type(y_predict))
y_predict = pd.DataFrame(y_predict)
y_predict.to_csv('../data/csv/submission_v3.csv', index=False)

# D_1 = model.predict([x1_prd,x2_prd])
# print("=====예상 값=====")
# print("D_1 : ",D_1)

plt.figure(figsize=(9,3.7))
plt.subplot(2,1,1)
plt.plot(y_predict, label='y_predict')
plt.legend()
