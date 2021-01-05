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

x = x.reshape(13,3,1)
print("x.shape : ", x.shape)  #(13,3,1)

# shuffle = 랜덤으로 섞음 , random_state (랜덤 난수 표 인덱스) @@@@@@@@@@@@ 필수 @@@@@@@@@@@@
x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )
x_train, x_test, y_train, y_test = train_test_split(
    x_train, x_train,train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )

#2. model config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM ,SimpleRNN

model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(SimpleRNN(10, activation='relu', input_length=3, input_dim=1))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
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