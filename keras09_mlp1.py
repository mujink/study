# 다 : 1
# np.transpose(x)...

import numpy as np

x = np.array([[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20]])
y = np.array([1,2,3,4,5,6,7,8,9,10])


print(x.shape)          #(2,10)

x = np.transpose(x)
print(x)
print(x.shape)          #(10,2)

#2. modelconfig
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_dim=2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. Compile
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x,y,epochs=100,batch_size=1,validation_split=0.2)

#4
loss, mae = model.evaluate(x,y)
print('loss : ', loss)
print('mae : ', mae)

y=predict = model.predict(x)
# print(y_predict)


# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE :", RMSE(y_test,y_predict))
# # print("mse : ", mean_squared_error(y_test, y_predict))
# print("mse : ", mean_squared_error(y_predict, y_test))

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print("R2 :",r2 )