# 1 : 다 앙상블을 구현하세요
# input range(1,3) output range(1,3)

import numpy as np

#1. data===============================================================
x1 = np.array([range(100), range(301,401), range(1,101)])

y1 = np.array([range(711,811), range(1,101), range(201,301)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)

y1 = np.transpose(y1)
y2 = np.transpose(y2)


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1, y2, shuffle=False, train_size=0.8
)


#2. model config===============================================================
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2.1 model1
input1 = Input(shape=(3,))
dense1 = Dense(3, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
dense1 = Dense(7, activation='relu')(dense1)
dense1 = Dense(9, activation='relu')(dense1)
dense1 = Dense(11)(dense1)
dense1 = Dense(17)(dense1)
dense1 = Dense(19)(dense1)

#2.4 model1 Branch of Output1
dense11 = Dense(21)(dense1)
dense11 = Dense(23)(dense11)
output1 = Dense(3)(dense11)

#2.5 model2 Branch of Output2
dense2 = Dense(21)(dense1)
dense2 = Dense(23)(dense2)
output2 = Dense(3)(dense1)

#2.6 def Model1,2
model = Model(inputs=input1, outputs=[output1, output2])
model.summary()

#3. Compilem,run===============================================================
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x1_train,[y1_train, y2_train], epochs=380, batch_size=1, validation_split=0.2, verbose=1)


# 4 Evaluation validation======================================================
loss = model.evaluate([x1_test, [y1_test, y2_test]], batch_size=1)
print('loss : ', loss)


# print("model.metrics_names :", Model.metrics_names)

y1_predict, y2_predict = model.predict(x1_test)
print("============================")
print("y1_predict \n:", y1_predict)
print("============================")
print("y2_predict \n:", y2_predict)
print("============================")
# RMSE...

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE :", (RMSE1 + RMSE2)/2)

# mse... 
MSE1 = mean_squared_error(y1_test, y1_predict)
MSE2 = mean_squared_error(y2_test, y2_predict)
print("mse : ", (MSE1+MSE2)/2 )

from sklearn.metrics import r2_score
r2_o1 = r2_score(y1_test, y1_predict)
r2_o2 = r2_score(y2_test, y2_predict)
print("R2 :", (r2_o1+r2_o2)/2 )