
import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense,LSTM,Conv1D,Conv2D,MaxPool1D,MaxPool2D,Dropout,Flatten, Input , Activation, Reshape, Lambda
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

x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x, y1 ,y2, train_size = 0.8, shuffle = True)#, random_state=1)
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
    model = Sequential()
    model.add(Conv1D(256,2,padding='same', activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(Conv1D(128,2,padding='same', activation='relu'))
    model.add(Conv1D(64,2,padding='same', activation='relu'))
    model.add(Conv1D(32,2,padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation= 'relu'))
    model.add(Dense(128, activation= 'relu'))
    model.add(Dense(64, activation= 'relu'))
    model.add(Dense(32, activation= 'relu'))
    model.add(Dense(16, activation= 'relu'))
    model.add(Dense(8, activation= 'relu'))
    model.add(Dense(4, activation= 'relu'))
    model.add(Dense(1, activation= 'relu'))
    model.summary()
    return model

# 퀀타일 로스 ============================================================================
import tensorflow.keras.backend as K

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

import tensorflow as tf

def quantile_loss2(y_true, y_pred):
    qs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)
# # # 콜백 ===============================================================================



# fitset = pd.read_csv('../data/csv/Dacon/preprocess_csv/TestDbSet2.csv', encoding='ms949', index_col=0)
fitset = np.load('../data/csv/Dacon/np/prdDb_X.npy',allow_pickle=True)
print(fitset.shape)

submission = pd.read_csv('../data/csv/Dacon/sample_submission.csv')

model = model1()
# model = load_model('../data/csv/Dacon/hdf5/Dacon_soler_cell_q0_"+ str(j) +"_x1.hdf5')


# 다음날 , 다 다음날 의 예상 치를 각각 핏함
# 분위수 별로 로스를 줄이기...
# 2일치씩 10분위로 하여 각각 하이퍼 파라미터를 조정하여야함
# 포문 안에 모델 세이브와 로드를 넣어 가중치를 업데이트 하는 방식으로 로스를 줄여봄
# 람다 함수 때문에 로드가 되지않음.
# 로드한 모델의 람다 함수를 찾을 수 없음 따로 파일을 생성해서 저장해야하는 것으로보임.
# 람다 함수로 넣지 않고 시도해봄 안됨.
# mse로 로스 줄여놓고 결과봄. 괜찮으면 서브미션 파일 생성할 때만 평가 진행함
# 결과확인

q = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# q = [0.1]

for j in q:
    # model = load_model("../data/csv/Dacon/hdf5/Dacon_soler_cell_q0_"+ str(j) +"_x1.hdf5")

    modelpath1 = "../data/csv/Dacon/hdf5/Dacon_soler_cell_q0_"+ str(j) +"_x1.hdf5"
    es1 = EarlyStopping(monitor = 'val_loss',patience=50, mode="min")
    cp1 = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath1, save_best_only=True, mode='min')
    reduce_lr1 = ReduceLROnPlateau(monitor='loss',patience=10, factor=0.7, verbose=1)

    optimizer = Adam(lr=0.002)
    optimizer1 = "adam"

    # model.compile(loss = "mse", optimizer = optimizer1)    
    model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = optimizer)    
    hist = model.fit (x_train, y1_train,  epochs= 1000, batch_size=1024, verbose=1, validation_data=(x_val , y1_val),callbacks=[cp1,es1,reduce_lr1], shuffle=True)
    temp = model.predict(fitset).round(2)
    col1 = 'q_' + str(j)
    print(temp.shape)
    print(temp)
    # temp = temp.reshape(temp.shape[0]*temp.shape[1],temp.shape[2])

    submission.loc[submission.id.str.contains("Day7"),col1] = temp[:,0]
    temp =[0]

# submission.to_csv('../data/csv/submission_v4_con1d4Test.csv', index=False)
# submission = pd.read_csv('../data/csv/submission_v4_con1d4Test.csv')

for j in q:
    # model = load_model("../data/csv/Dacon/hdf5/Dacon_soler_cell_q0_"+ str(j) +"_x2.hdf5")

    modelpath2 = "../data/csv/Dacon/hdf5/Dacon_soler_cell_q0_"+ str(j) +"_x2.hdf5"
    es2 = EarlyStopping(monitor = 'val_loss',patience=50, mode="min")
    cp2 = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath2, save_best_only=True, mode='min')
    reduce_lr2 = ReduceLROnPlateau(monitor='loss',patience=10, factor=0.7, verbose=1)

    optimizer = Adam(lr=0.002)
    optimizer1 = "adam"

    # model.compile(loss = "mse", optimizer = optimizer1 )    
    model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = optimizer)
    hist = model.fit (x_train, y2_train,  epochs= 1000, batch_size=1024, verbose=1, validation_data=(x_val , y2_val),callbacks=[cp2,es2,reduce_lr1], shuffle=True)
    temp = model.predict(fitset).round(2)
    col2 = 'q_' + str(j)
    print(temp.shape)
    print(temp)
    # temp = temp.reshape(temp.shape[0]*temp.shape[1],temp.shape[2])
    temp[temp<0] = 0

    submission.loc[submission.id.str.contains("Day8"),col2] = temp[:,0]
    temp =[0]

submission.to_csv('../data/csv/submission_v4_con1d4Test.csv', index=False)

"""
1.4529
2.3290
2.7596
2.8833
2.7781
2.5062
2.0878
1.5475
0.8759

1.4653
2.4096
2.9216
3.0697
2.9473
2.6341
2.1894
1.6066
0.9067

큰 차이없음
1.4214
2.2239
2.6119
2.6630
2.5241
2.2378
1.8197
1.3015
0.7470

1.3964
2.2243
2.6067
2.6673
2.4960
2.1947
1.7732
1.2704
0.7140

에코 높임 1000 얼리스탑 따위는 없음
의미 없는 그냥 과적합임
과적합 가즈아아아아 결과로 보고 판단해줌
안 끝나니? 퇴근 안할꺼니?
집에 보내줘. 구해줘. 구해줘 갇혓어

과적합 폭망

다시할거야
"""
