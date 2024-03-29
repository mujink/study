# 다 : 1
# 실습 트래인 과 테스트 분리 알2 알엠에스이
# random_state....range(1,3)

import numpy as np

#1 data
x = np.array([range(100), range(301,401), range(1,101)])
y = np.array(range(711,811))
print(x.shape)          #(3,100)
print(y.shape)          #(100,)
 
x = np.transpose(x)
# print(x)
print(x.shape)          #(100,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )

#2. modelconfig
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# from keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_dim=3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. Compilem,run
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train,y_train,epochs=100,batch_size=1,validation_split=0.2)

#4 Evaluation validation
loss, mae = model.evaluate(x_train,y_train)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)
# print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE :", RMSE(y_test,y_predict))
# print("mse : ", mean_squared_error(x_test, y_test))
print("mse : ", mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 :",r2 )
