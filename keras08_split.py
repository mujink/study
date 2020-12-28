from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

# np.array()
# array()

#1. 데이터

x = np.array(range(1,101))
y = np.array(range(101,201))

# 리스트의 슬라이싱
x_train = x[0:60]   #  0~59 번쨰 까지 :::: 값 1~60
x_val = x[60:80]    #  61~80
x_test = x[80:]     #  81 ~ 100

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2)

#4. 평가 예측
results = model.evaluate(x_test, y_test, batch_size=1)
print ("mse, mae : ", results)

y_predict = model.predict(x_test)
# print("y_predict : ", y_pred)
# print("y_predict : ", y_predict)

#. 사이킷 런이 무엇인가
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    # RMES => np.sqrt => 두 리스트의 제곱 오차 평균
print("RMSE :", RMSE(y_test,y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 :",r2 )