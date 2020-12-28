# 네이밍 룰
import numpy as np
import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layer import model

#1. 데이터
#... test 데이터 추가하여 모델평가.
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
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(X_test, y_test, batch_size=1)
print('loss :', loss)

result = model.predict([9])
print('result : ', result)
