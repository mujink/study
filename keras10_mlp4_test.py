# 다 : 다
#실습
# x는 (100,5) 데이터 구성
# y는 (100,2) 데이터 구성
# 모델을 완성하세요.

# 다 만든 사람 predict의 일부 값을 출력

import numpy as np

#1. data
x = np.array([range(100), range(301,401), range(1,101), range(200,300), range(401,501)])
y = np.array([range(711,811), range(201,301), ])

x_pred2 = np.array([100,401,101,300,501])

print(x.shape)          #(3,100)
print(y.shape)          #(3,100)
 
x = np.transpose(x)
y = np.transpose(y)

x_pred2 = x_pred2.reshape(1,5)

print(x.shape)          #(100,3)
print(y.shape)          #(100,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )
print(x_train.shape)    #(80,5)
print(y_train.shape)    #(80,2)

#2. model config

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# from keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_dim=5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

#3. Compilem,run
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2)

#4 Evaluation validation
loss, mae = model.evaluate(x_train,y_train)
print('loss : ', loss)
print('mae : ', mae)


y_predict = model.predict(x_test)
print("y_predict : ",  y_predict)
print("y_predict : ",  y_predict[10:11])

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE :", RMSE(y_test,y_predict))
# print("mse : ", mean_squared_error(x_test, y_test))
print("mse : ", mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 :",r2 )

y_Pred2 = model.predict(x_pred2)
print("y_pred2 : " , y_Pred2)
# [[811.00024 301.0001 ]]