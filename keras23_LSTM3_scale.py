# 코딩하시오!! LSTM
# 나는 80을 원하고 있다.

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

x = x.reshape(13,3,1)
print("x.shape : ", x.shape)  #(13,3,1)


#2. model config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='linear', input_shape=(3,1)))
model.add(Dense(50, activation='linear'))
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
"""