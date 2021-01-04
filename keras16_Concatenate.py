# concatenate => Concatenate
# input range(1,3) output range(1,3)

import numpy as np

#1. data===============================================================
x1 = np.array([range(100), range(301,401), range(1,101)])
y1 = np.array([range(711,811), range(1,101), range(201,301)])

x2 = np.array([range(101, 201), range(411,511), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

print(x1.shape)          #(100,3)
print(y1.shape)          #(100,3)
print(x2.shape)          #(100,3)
print(y2.shape)          #(100,3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, shuffle=False, train_size=0.8
)
print(x1_train.shape)          #(100,3)
print(y1_train.shape)          #(100,3)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, shuffle=False, train_size=0.8
)
print(x2_train.shape)          #(100,3)
print(y2_train.shape)          #(100,3)




#2. model config===============================================================
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2.1 model1
input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
# output1 = Dense(3)(dense1)

#2.2 model2
input2 = Input(shape=(3,))
dense2 = Dense(5, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
# output2 = Dense(3)(dense2)

#2.3 model Concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate
# from keras.layers import concatenate

merge1 = Concatenate(axis=1)([dense1, dense2])
middle1 = Dense(3)(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(9)(middle1)

#2.4 model1 Branch of Output1
output1 = Dense(11)(middle1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1)

#2.5 model2 Branch of Output2
output2 = Dense(15)(middle1)
output2 = Dense(10)(output2)
output2 = Dense(10)(output2)
output2 = Dense(3)(output2)

#2.6 def Model1,2
model = Model(inputs=[input1, input2],outputs=[output1, output2])
model.summary()

#3. Compilem,run===============================================================
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],[y1_train, y2_train], epochs=380, batch_size=1, validation_split=0.2, verbose=1)


# 4 Evaluation validation======================================================
loss = model.evaluate([x1_test, y1_test], [x2_test, y2_test], batch_size=1)
print('loss : ', loss)


# print("model.metrics_names :", Model.metrics_names)

y1_predict, y2_predict = model.predict([x1_test,x2_test])
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
print("RMSE :", (RMSE1 + RMSE2)/2 )

# mse... 
MSE1 = mean_squared_error(y1_test, y1_predict)
MSE2 = mean_squared_error(y2_test, y2_predict)
print("mse : ", (MSE1 + MSE2)/2 )

from sklearn.metrics import r2_score
r2_m1 = r2_score(y1_test, y1_predict)
r2_m2 = r2_score(y2_test, y2_predict)
print("R2 :",(r2_m1 + r2_m2)/2 )