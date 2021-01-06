#  keras23_3을 카피해서
#  LSTM층을 두개를 만들 것!!

#1. data

import numpy as np
# x = np.array([range(3), range(3), range(3)])

x= np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],[20,30,40],
            [30,40,50],[40,50,60]])
y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


print("x.shape : ", x.shape)  #(13,3)
print("y.shape : ", y.shape)  #(13,)

#1.1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler @@@@@@@@@@@@ 필수 @@@@@@@@@@@@

scaler = MinMaxScaler()
scaler.fit(x)
x_train = scaler.transform(x)

x = x.reshape(x.shape[0],x.shape[1],1)

# x = x.reshape(13,3,1)
# print("x.shape : ", x.shape)  #(13,3,1)

# shuffle = 랜덤으로 섞음 , random_state (랜덤 난수 표 인덱스) @@@@@@@@@@@@ 필수 @@@@@@@@@@@@
x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )
x_train, x_val, y_train, y_val = train_test_split(
    x_train, x_train,train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )


#2. model config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='linear', input_shape=(3,1), return_sequences=True))
model.add(LSTM(10, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(1, activation='linear'))

model.summary()

#3. Compile, train

# from tensorflow.keras.callbacks import EarlyStopping
#                 # mean_squared_error
# early_stopping = EarlyStopping(monitor='loss', patience=30, mode='min') 

model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=500, batch_size=1, verbose=1, callbacks=[early_stopping])
model.fit(x, y, epochs=500, batch_size=1, verbose=1)

#4. Evaluate, Predict
loss = model.evaluate(x,y)
print("loss : ", loss)

x_pred = np.array([50,60,70])
x_pred = x_pred.reshape(1,3,1)

y_pred = model.predict(x_pred)
print(y_pred)

"""
 80+1(????)

loss :  0.014979862608015537
[[80.9897]]

loss :  0.03203417360782623
[[81.34958]]

loss :  0.15643206238746643
[[81.06115]]

all activation='linear'
loss :  4.84839292766992e-05
[[80.84219]]

loss :  0.006368533242493868
[[81.651245]]

callback = 30
loss :  0.11684146523475647
[[80.10127]]

loss :  0.2963032126426697
[[79.994774]]

callback = 50
loss :  0.011367928236722946
[[79.93723]]

loss :  0.022982282564044
[[81.254]]


callbacks = off
All layeractivation = 'linear'
loss :  0.017403127625584602
[[79.66048]]

loss :  0.004742923192679882
[[80.22905]]

loss :  0.007386711426079273
[[78.70087]]

loss :  0.08146253228187561
[[81.05565]]

loss :  0.1900707334280014
[[81.0779]]


LSTM 1,2Layer
loss :  0.008844711817800999
[[80.63693]]

loss :  0.8462216258049011
[[76.92895]]

loss :  0.7550256252288818
[[81.38269]]

loss :  0.9194720387458801
[[79.07298]]

loss :  0.02087796851992607
[[81.74035]]

loss :  0.0014255426358431578
[[81.38179]]

loss :  0.023815380409359932
[[80.843475]]

LSTM 1,2,3Layer
loss :  0.00891756359487772
[[80.99793]]

loss :  0.020112784579396248
[[81.03767]]

loss :  0.020112784579396248
[[81.03767]]

loss :  0.0025026516523212194
[[80.641556]]
"""