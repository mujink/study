#... @@@@ 외워 머신러닝 딥러닝 기본 프레임워크
import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델 구성 (순차모델 레이어 구성)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear'))
model.add(Dense(3, activation='linear'))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y,batch_size=1)
print("loss : ", loss)

result = model.predict ([4])
print('result :', result)