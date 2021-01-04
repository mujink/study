#  실습
#  R2를 음수가 아닌 0.5 이하로 줄이기
#  1. 레이어는 인풋과 아웃풋을 포함 6개 이상
#  2. batch_size = 1
#  3. epochs = 100 이상
#  4. 히든레이어의 노드의 갯수는 10 이상
#  5. 데이터 조작 금지.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

# np.array()
# array()

#1. 데이터

x_trian = np.array([1,2,3,4,5,6,7,8,9,10])
y_trian = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])

x_Pred = array([16,17,18])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam', metrics=['mae'])
model.fit(x_trian, y_trian, epochs=300, batch_size=1, validation_split=0.1)

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

print("RMSE :", RMSE(y_test,y_predict))

# print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("R2 :",r2 )