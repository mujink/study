# 열이 다른 앙상블

#1. data

import numpy as np
from numpy import array
# x = np.array([range(3), range(3), range(3)])

x1= np.array([[1,2],[2,3],[3,4],[4,5],
            [5,6],[6,7],[7,8],[8,9],
            [9,10],[10,11],
            [20,30],[30,40],[40,50]])
x2= np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]])
y1 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]])
y2= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x1_predict = array([55,65])  #(2,)
x2_predict = array([65,75,85])  #(3,)

print("x1.shape : ", x1.shape)  #(13,2)
print("x2.shape : ", x2.shape)  #(13,3)
print("y1.shape : ", y1.shape)  #(13,3)
print("y2.shape : ", y2.shape)  #(13,)

#1.1
from sklearn.model_selection import train_test_split

x1_predict = x1_predict.reshape(1,2) #(3,) => (1,2)
x2_predict = x2_predict.reshape(1,3) #(3,) => (1,3)

y1 = y1.reshape(13,3) #(13,3) => (13,3)
y2 = y2.reshape(13,1) #(13,1) => (13,1)

x1 = x1.reshape(x1.shape[0],x1.shape[1],1) #(13,2) => (13,2,1)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1) #(13,3) => (13,3,1)

x1_predict = x1_predict.reshape(x1_predict.shape[0],x1_predict.shape[1],1) #(1,2) => (1,2,1)
print("x1_predict.shape : ", x1_predict.shape)  
x2_predict = x2_predict.reshape(x2_predict.shape[0],x2_predict.shape[1],1) #(1,3) => (1,3,1)
print("x2_predict.shape : ", x2_predict.shape)

print("x1.shape : ",x1.shape) #(13,2,1)
print("x2.shape : ",x2.shape) #(13,3,1)

#2. model config
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, LSTM, Input


input1 = Input(shape=(2,1))
Hlayer11 = LSTM(100, activation='relu')(input1)
Hlayer12 = Dense(30, activation='relu')(Hlayer11)
Hlayer13 = Dense(40, activation='relu')(Hlayer12)
Hlayer14 = Dense(40, activation='relu')(Hlayer13)
Hlayer15 = Dense(40, activation='relu')(Hlayer14)

input2 = Input(shape=(3,1))
Hlayer21 = LSTM(100, activation='relu')(input1)
Hlayer22 = Dense(30, activation='relu')(Hlayer21)
Hlayer23 = Dense(40, activation='relu')(Hlayer22)
Hlayer24 = Dense(40, activation='relu')(Hlayer23)
Hlayer25 = Dense(40, activation='relu')(Hlayer24)

#2.3 model Concatenate
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = Concatenate(axis=1)([Hlayer15, Hlayer25])

#2.4 model1 Branch of Output1
output1 = Dense(50,activation='relu')(merge1)
output1 = Dense(30,activation='relu')(output1)
output1 = Dense(3)(output1)

#2.5 model2 Branch of Output2
output2 = Dense(50,activation='relu')(merge1)
output2 = Dense(30,activation='relu')(output2)
output2 = Dense(1)(output2)

#2.6 def Model1,2
inputs = [input1, input2]
outputs = [output1, output2]

model = Model(inputs =  inputs, outputs = outputs)
model.summary()
x = [x1,x2]
y = [y1,y2]
x_predict = [x1_predict, x2_predict]
# 3. Compile, train

from tensorflow.keras.callbacks import EarlyStopping
                # mean_squared_error
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='min') 



model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1,
            # verbose=1, callbacks=[early_stopping])
            verbose=1)
# model.fit([x1_train,x2_train], y_train, epochs=500, batch_size=1, verbose=1)

# #4. Evaluate, Predict
loss = model.evaluate(x,y)
y_pred = model.predict(x_predict)
print("loss , y_pred: ", loss, y_pred)

# from sklearn.metrics import r2_score
# r2_m1 = r2_score(y, y_pred)
# print("R2 :", r2_m1 )

# print("y_pred : " , y_Pred)
"""
loss :  [0.3845544457435608, 0.36615151166915894, 0.018402928486466408]
[array([[6.044095 , 6.897272 , 7.3191953]], dtype=float32), array([[88.1528]], dtype=float32)]

loss :  [0.3968242108821869, 0.24738186597824097, 0.14944234490394592]
[array([[5.3325305, 6.2020946, 7.4294643]], dtype=float32), array([[87.88725]], dtype=float32)]
"""