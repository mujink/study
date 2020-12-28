# 네이밍 룰
import numpy as np
import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
#... test 데이터 추가하여 모델평가. => loss = 0에 근접하기
X_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])
X_test = np.array([1,2,3,4,5,6,7,8,9,10])
y_test = np.array([2,3,4,5,6,7,8,9,10,11])

X_predict = np.array([111,112,113])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(4,activation='linear'))
model.add(Dense(10,activation='relu'))
model.add(Dense(3,activation='linear'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(X_test, y_test,batch_size=1)
print('loss : ', loss)

result = model.predict(X_predict)
print('result : ', result)
