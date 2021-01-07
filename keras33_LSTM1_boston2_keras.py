#  텐서플로 데이터셋
# LSTM 모델링
#  덴스와 성능 비교
# 회귀

#1. DATA
from tensorflow.keras.datasets import boston_housing
import numpy as np

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print(x_train.shape) #(404, 13)

#1.1 Data Preprocessing
from sklearn.model_selection import train_test_split

x_train, x_validation, y_train, y_validation = train_test_split(
    x_train, y_train, train_size = 0.8, shuffle = True, random_state=66)

print(x_train.shape)    #(323, 13)
print(x_test.shape)     #(102, 13)
print(np.min(x_train), np.max(x_train)) # 0.0 ~ 711.0

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],1)

#2. Modeling
# print(boston_housing.DESCR)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , LSTM

model = Sequential()
model.add(LSTM(50, input_shape=(13,1), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=50, mode='min')

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train, y_train, epochs=500, batch_size=10, validation_data=(x_validation, y_validation),
    callbacks=(early_stopping), verbose=1)

# 4 Evaluation validation
loss, mae = model.evaluate(x_test, y_test, batch_size=10)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


# Dense model
"""
loss :  11.298630714416504
mae :  2.318967342376709
RMSE :  3.3613434376432876
R2 :  0.8642706949919532
"""
# LSTM model
"""
loss :  20.14045524597168
mae :  2.996915102005005
RMSE :  4.487811880539783
R2 :  0.7580547291692853
"""