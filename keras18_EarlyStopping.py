# EarlyStopping : 조건에 맞으면 epochs를 멈춤
# model.fit에서 실행
# #1. Data
import numpy as np
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape) #(506, 13)
print(y.shape) #(506,)
print('====================')
print(x[:5])
print(y[:10])

print(np.max(x[0]))

# 1.1 Data Preprocessing
# MinMaxScaler 필수


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=33
)

from sklearn.model_selection import train_test_split
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


# 2. model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


input1 = Input(shape=(13,))
d1 = Dense(1000, activation='relu')(input1)
dh = Dense(200, activation='relu')(d1)
dh = Dense(200, activation='relu')(d1)
dh = Dense(300, activation='relu')(d1)
dh = Dense(300, activation='relu')(dh)
outputs = Dense(1)(dh)

# 2.1 model def
model = Model(inputs =  input1, outputs = outputs)
model.summary()

# 3. Compile, run
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='min')

model.fit(
    x_train,y_train,
    epochs=2000, batch_size= 8, validation_data=(x_val, y_val), verbose=2, callbacks=[early_stopping])

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

# 제대로 전처리 (validation_data)
# 결과 값
"""
loss :  10.806500434875488
RMSE : 3.2873240846554324
RMSE : 10.806499637555676
R2 : 0.849143856752024

loss :  2.1226985454559326
RMSE : 3.7007119725686497
RMSE : 13.695269103912945
R2 : 0.8088173278071026
"""