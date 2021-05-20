
import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense,LSTM,Conv1D,MaxPool1D,Dropout,Flatten, Input , Activation, Reshape, Lambda
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


#  넘파이 불러오기=======================================================

x = np.load('../data/csv/Dacon/np/TrainDb_X.npy',allow_pickle=True)
y = np.load('../data/csv/Dacon/np/TrainDb_Y.npy',allow_pickle=True)

print(x.shape)
print(y.shape)
"""
(52464, 6)
(52464, 2)
"""
# x = x.reshape(-1,48,10).astype('float32')
# y = y.reshape(-1,48,2).astype('float32')
print(x.shape)
print(y.shape)
"""
(1093, 48, 6)
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
(33576, 6)
(10493, 6)
(8395, 6)
(33576, 2)
(10493, 2)
(8395, 2)
"""

# x_train = x_train.reshape()
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Adam(lr=0.1)
# loss :  0.8494184613227844 y_pred :  [[12.966707]]

# optimizer = Adam(lr=0.01)
# loss :  0.009967544116079807 y_pred :  [[11.189517]]

optimizer = Adam(lr=0.0005)
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

# optimizer = Adagrad(lr=0.01)
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

# optimizer = Nadam(lr=0.1)
# loss :  12.602835655212402 y_pred :  [[15.236937]]

# optimizer = Nadam(lr=0.01)
# loss :  9.379164112033322e-13 y_pred :  [[11.000004]] @@@@@@@@@@@@@@@@

# optimizer = Nadam(lr=0.001)
# loss :  2.149036591042597e-12 y_pred :  [[11.000001]] 

# optimizer = Nadam(lr=0.0001)
# loss :  5.9579133449005894e-06 y_pred :  [[10.995723]]
# ============================================================



# 모델===============================================================================
# def model() :
input1 = Input(shape=(x_train.shape[1]))
output1 = Dense(256, activation= 'relu')(input1)
output1 = Dense(128, activation= 'relu')(output1)
output1 = Dense(64, activation= 'relu')(output1)
output1 = Dense(16, activation= 'relu')(output1)
output1 = Dense(8, activation= 'relu')(output1)
output1 = Dense(2, activation= 'relu')(output1)

    # 2.6 def Model1,2
model = Model(inputs=input1, outputs=output1)
model.summary()

import tensorflow.keras.backend as K

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

# # # 콜백 ===============================================================================
modelpath = "../data/h5/Dacon_soler_cell.hdf5"
es = EarlyStopping(monitor = 'loss',patience=100, mode="min")
cp = ModelCheckpoint(monitor = 'loss',filepath = modelpath, save_best_only=True, mode='min')
# reduce_lr = ReduceLROnPlateau(monitor='loss',patience=10, factor=0.3, verbose=1)


fitset = pd.read_csv('../data/csv/Dacon/preprocess_csv/TestDbSet2.csv', encoding='ms949', index_col=0)
submission = pd.read_csv('../data/csv/Dacon/sample_submission.csv')

q = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for j in q:
    # model = model()
    # model = load_model("../data/h5/Dacon_soler_cell.hdf5")
    model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = optimizer, metrics=['mae'])
    hist = model.fit (x_train, y_train,  epochs= 10000, batch_size=128, verbose=1, validation_data=(x_val , y_val),callbacks=[es,cp], shuffle=False)
    temp = model.predict(fitset)
    print(temp)
    col = 'q_' + str(j)
    submission.loc[submission.id.str.contains("Day7"),col] = temp[:,0]
    submission.loc[submission.id.str.contains("Day8"),col] = temp[:,1]

submission.to_csv('../data/csv/submission_v4_2.csv', index=False)

#  모델 불러오기
# model = load_model('../data/h5/Dacon_soler_cell.hdf5')

loss = model.evaluate(x_test, y_test, batch_size=128)
print("loss : ",loss)

y_predict = model.predict(x_test)

# print(y_test.shape)
# print(y_predict.shape)
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)

# print(y_predict)
# print(type(y_predict))
# y_predict = pd.DataFrame(y_predict)
# y_predict.to_csv('../data/csv/submission_v3.csv', index=False)

# plt.figure(figsize=(9,3.7))
# plt.subplot(2,1,1)
# plt.plot(y_predict, label='y_predict')
# plt.legend()




# 멈춰라 얍 멈춰라 얍 뭔가 문제가 있는거 아냐?
