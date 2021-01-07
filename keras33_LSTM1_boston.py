#  사이킷 런
# LSTM 모델링
#  덴스와 성능 비교
# 회귀

#1. Data
import numpy as np
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape) #(506, 13)
print(y.shape) #(506,)
print('====================')


# 1.1 Data Preprocessing
# MinMaxScaler 필수


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=33
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,  test_size=0.2
)

# x의 값들을 0~1 사이 값으로 줄임 => 가중치를 낮추어 연산 속도가 빨라짐.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# print('====================')
# x = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
# x = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
# x = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
# print("x_train.shape : " , x_train.shape) #(323, 13, 1)
# print("x_test.shape : " , x_test.shape) #(102, 13, 1)
# print("x_val.shape : " , x_val.shape) #(81, 13, 1)
# print('====================')
# 2. model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM


input1 = Input(shape=(13,))
d1 = Dense(30, activation='relu')(input1)
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

\