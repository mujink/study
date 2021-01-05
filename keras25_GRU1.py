# Keras23_LSTM1 로 GRU 코드 완성
# GRU 파라미터 분석할 것
# 결과치 Loss와 predict값 LSTM과 비교

#1. data

import numpy as np
# x = np.array([range(3), range(3), range(3)])

x= np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y= np.array([4,5,6,7])

print("x.shape : ", x.shape)  #(4,3)
print("y.shape : ", y.shape)  #(4,)

x = x.reshape(4,3,1)
print("x.shape : ", x.shape)  #(4,3,1)


#2. model config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM ,SimpleRNN, GRU

model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(GRU(10, activation='linear', input_length=3, input_dim=1))
model.add(Dense(50, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(1, activation='linear'))

model.summary()
#3. Compile, train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1, verbose=1)

#4. Evaluate, Predict
loss = model.evaluate(x,y)
print("loss : ", loss)

x_pred = np.array([5,6,7]) #(3,1) -> (1,3,1)
x_pred = x_pred.reshape(1,3,1)

y_pred = model.predict(x_pred)
print(y_pred)

"""
loss :  0.0035664266906678677
[[7.958768]]

loss :  0.006438890006393194
[[8.155805]]

All layer activation='linear'
loss :  0.0004961384693160653
[[7.9067545]]

loss :  0.004142099525779486
[[7.8085036]]

loss :  0.00744429137557745
[[7.624658]]

loss :  0.0028342092409729958
[[7.8135624]]

epochs=500
loss :  0.0001339292648481205
[[8.02248]]

loss :  0.0003078212321270257
[[7.9205766]]

loss :  6.304095586529002e-05
[[8.004461]]
"""