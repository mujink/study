# keras23_LSTM3_scale 을 DNN으로 코딩
# 결과치 비교
# DNN으로 23번 파일보다 loss를 좋게 만들것

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

# x = x.reshape(13,3,1)
# print("x.shape : ", x.shape)  #(13,3,1)

#2. model config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(100, activation='linear', input_shape=(3,)))
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

x_pred = np.array([[50,60,70]])
# x_pred = x_pred.reshape(1,3,1)

y_Pred = model.predict(x_pred)
print("y_pred : " , y_Pred)
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
"""





"""
DNN
loss :  0.00036582184839062393
y_pred :  [[79.98117]]

loss :  0.2476692944765091
y_pred :  [[78.61985]]

loss :  0.5798483490943909
y_pred :  [[79.47507]]




DNN ()@@@@@@@@@@ 
activation='linear'  더 잘나옴..
loss :  3.1360077729081226e-11
y_pred :  [[79.99998]]

loss :  2.4306245904881507e-10
y_pred :  [[79.99999]]

loss :  0.1604553908109665
y_pred :  [[81.151665]]

loss :  1.052915046817482e-11
y_pred :  [[80.00001]]

loss :  4.122459665301115e-11
y_pred :  [[80.00002]]
"""