
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
y1 = np.load('../data/csv/Dacon/np/TrainDb_Y1.npy',allow_pickle=True)
y2 = np.load('../data/csv/Dacon/np/TrainDb_Y2.npy',allow_pickle=True)

print(x.shape)
print(y1.shape)
print(y2.shape)
"""
(52464, ?)
(52464, 2)
"""
# x = x.reshape(-1,48,?).astype('float32')
# y = y.reshape(-1,48,2).astype('float32')
print(x.shape)
print(y1.shape)
print(y2.shape)
"""
(1093, 48, 6)
(1093, 48, 2)
"""
# train_test_split ==========================================================================================
from sklearn.model_selection import train_test_split

x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x, y1 ,y2, train_size = 0.8, shuffle = False)#, random_state=1)
print("# shape  test=====================================================================================================")
print(x_train.shape)
print(x_val.shape)

print(y1_train.shape)
print(y1_val.shape)

print(y2_train.shape)
print(y2_val.shape)

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

# optimizer = Adam(lr=0.006)
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
def model1() :

    input1 = Input(shape=(x_train.shape[1],x_train.shape[2]))
    con1 = Conv1D(256,2,padding='same', activation='relu')(input1)
    # MxP1 = MaxPool1D(pool_size=(2))(con1)
    con1 = Conv1D(128,2,padding='same', activation='relu')(con1)
    # MxP1 = MaxPool1D(pool_size=(2))(con1)
    con1 = Conv1D(64,2,padding='same', activation='relu')(con1)
    con1 = Conv1D(32,2,padding='same', activation='relu')(con1)
    # MxP1 = MaxPool1D(pool_size=(2))(con1)
    # con1 = Conv1D(16,2,padding='same', activation='relu')(MxP1)
    # MxP1 = MaxPool1D(pool_size=(2))(con1)
    output1 = Flatten()(con1)
    output1 = Dense(256, activation= 'relu')(output1)
    output1 = Dense(128, activation= 'relu')(output1)
    output1 = Dense(64, activation= 'relu')(output1)
    output1 = Dense(32, activation= 'relu')(output1)
    output1 = Dense(16, activation= 'relu')(output1)
    output1 = Dense(8, activation= 'relu')(output1)
    output1 = Dense(4, activation= 'relu')(output1)
    output1 = Dense(2, activation= 'relu')(output1)

    # 2.6 def Model1,2
    model = Model(inputs=input1, outputs=output1)
    model.summary()
    return model

import tensorflow.keras.backend as K

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

# # # 콜백 ===============================================================================
modelpath = "../data/h5/Dacon_soler_cell.hdf5"
es = EarlyStopping(monitor = 'val_loss',patience=30, mode="min")
cp = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath, save_best_only=True, mode='min')
# reduce_lr = ReduceLROnPlateau(monitor='loss',patience=15, factor=0.9, verbose=1)


# fitset = pd.read_csv('../data/csv/Dacon/preprocess_csv/TestDbSet2.csv', encoding='ms949', index_col=0)
fitset = np.load('../data/csv/Dacon/np/prdDb_X.npy',allow_pickle=True)
print(fitset.shape)

submission = pd.read_csv('../data/csv/Dacon/sample_submission.csv')

model = model1()
# model = load_model('../data/h5/Dacon_soler_cell.hdf5')

q = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for j in q:
    optimizer = Adam(lr=0.0018)
    model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = optimizer)
    hist = model.fit (x_train, y1_train,  epochs= 10000, batch_size=256, verbose=1, validation_data=(x_val , y1_val),callbacks=[es,cp], shuffle=False)
    temp = model.predict(fitset)
    col = 'q_' + str(j)
    print(temp.shape)
    print(temp)
    # temp = temp.reshape(temp.shape[0]*temp.shape[1],temp.shape[2])
    submission.loc[submission.id.str.contains("Day7"),col] = temp[:,0]
    temp =[0]
    
q = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for j in q:
    optimizer = Adam(lr=0.0018)
    model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = optimizer)
    hist = model.fit (x_train, y2_train,  epochs= 10000, batch_size=256, verbose=1, validation_data=(x_val , y1_val),callbacks=[es,cp], shuffle=False)
    temp = model.predict(fitset)
    col = 'q_' + str(j)
    print(temp.shape)
    print(temp)
    # temp = temp.reshape(temp.shape[0]*temp.shape[1],temp.shape[2])
    submission.loc[submission.id.str.contains("Day8"),col] = temp[:,1]
    temp =[0]

submission.to_csv('../data/csv/submission_v4_con1d2.csv', index=False)

#  모델 불러오기
# model = load_model('../data/h5/Dacon_soler_cell.hdf5')

# loss = model.evaluate(x_test, y_test, batch_size=256)
# print("loss : ",loss)

# y_predict = model.predict(x_test)

# print(y_predict)
# print(type(y_predict))
# y_predict = pd.DataFrame(y_predict)
# y_predict.to_csv('../data/csv/submission_v3.csv', index=False)

# plt.figure(figsize=(9,3.7))
# plt.subplot(2,1,1)
# plt.plot(y_predict, label='y_predict')
# plt.legend()




# 멈춰라 얍 멈춰라 얍 뭔가 문제가 있는거 아냐?