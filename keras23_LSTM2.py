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
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(LSTM(10, activation='linear', input_length=3, input_dim=1))
model.add(Dense(50, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(1, activation='linear'))

model.summary()
#3. Compile, train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, verbose=1)

#4. Evaluate, Predict
loss = model.evaluate(x,y)
print("loss : ", loss)

x_pred = np.array([5,6,7]) #(3,1) -> (1,3,1)
x_pred = x_pred.reshape(1,3,1)

y_pred = model.predict(x_pred)
print(y_pred)