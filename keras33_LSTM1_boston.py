#  사이킷 런
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


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM


input1 = Input(shape=(13,))
d1 = LSTM(30, activation='relu')(input1)
dh = Dense(50, activation='relu')(d1)
dh = Dense(100, activation='relu')(dh)
dh = Dense(100, activation='relu')(dh)
dh = Dense(30, activation='relu')(dh)
outputs = Dense(1)(dh)

# 2.1 model def
model = Model(inputs =  input1, outputs = outputs)
model.summary()

# 3. Compile, run
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
                # mean_squared_error
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='min')

model.fit(
    x_train,y_train,
    epochs=1000, batch_size=1, validation_data=(x_val, y_val), verbose=1, callbacks=(early_stopping))

# 4 Evaluation validation
loss, mae= model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE :", RMSE(y_test, y_predict) )

# mse... 
MSE1 = mean_squared_error(y_test, y_predict)

print("RMSE :", MSE1 )

from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_predict)

print("R2 :", r2_m1 )

