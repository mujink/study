#  2개의 파일을 만드시오.
#  1. EarlyStopping 을 적용하지 않은 최고의 모델
#  2. EarlyStopping 을 적용한 최고의 모델


#1. DATA
from tensorflow.keras.datasets import boston_housing
import numpy as np

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

#1.1 Data Preprocessing
from sklearn.model_selection import train_test_split

x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=66)

# print(x_train.shape) #(404, 13)
# print(np.min(x_train), np.max(x_train)) # 0.0 ~ 711.0
print(x_train.shape)    #(323, 13)
print(x_test.shape)     #(102, 13)
print(np.min(x_train), np.max(x_train)) # 0.0 ~ 711.0

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

#2. Modeling
# print(boston_housing.DESCR)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(50, input_dim=13, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience=30, mode='min')

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train, y_train, epochs=500, batch_size=10, validation_data=(x_validation, y_validation), verbose=1)

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



"""
loss :  11.298630714416504
mae :  2.318967342376709
RMSE :  3.3613434376432876
R2 :  0.8642706949919532
"""