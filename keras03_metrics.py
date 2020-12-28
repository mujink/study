# 네이밍 룰
import numpy as np
import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layer import model

#1. 데이터
X_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
X_test = np.array([6,7,8])
y_test = np.array([6,7,8])

#2. 모델구성
model = Sequential()
# model = Keras.model Sequential()
model.add(Dense(5, input_dim=1, activation='linear'))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
#...@@@@ metrice =['accuracy'] 학습(?)
#...@@@@ metrice =['mse'] 평균 제곱 오차(?)
#...@@@@ metrice =['mae'] 평균 절대 오차(?)
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(X_test, y_test, batch_size=1)
print('loss :', loss)

# result = model.predict([9])
result = model.predict([X_train])
print('result : ', result)
