from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터

x = np.array(range(1,101))
y = np.array(range(1,101))

x_train = x[:60]   #  0~59 번쨰 까지 :::: 값 1~60
x_val = x[60:80]    #  61~80
x_test = x[80:]     #  81 ~ 100
 
y_train = y[:60]    #  0~59 번쨰 까지 :::: 값 1~60
y_val = x[60:80]    #  61~80
y_test = x[80:]     #  81 ~ 100

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True)
print (x_train)
print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)


#4. 평가예측
loss, mse = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('mse : ', mse)

y_predict = model.predict(x_test)
print(y_predict)

#shuffle = False
# loss :  1.395869730913546e-05
# mse :  0.0037330626510083675

#shuffle = True
# loss :  0.00032849906710907817
# mse :  0.015596771612763405