# 행이 다른 앙상블

#1. data

import numpy as np
from numpy import array
# x = np.array([range(3), range(3), range(3)])

x1= np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12]])
            # [20,30,40],[30,40,50],[40,50,60]])
x2= np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]])
y1= np.array([4,5,6,7,8,9,10,11,12,13])
            # ,50,60,70])
y2= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x1_predict = array([55,65,75])  #(3,)
x2_predict = array([65,75,85])  #(3,)

print("x1.shape : ", x1.shape)  #(10,3)
print("x2.shape : ", x2.shape)  #(13,3)
print("y1.shape : ", y1.shape)  #(10,)
print("y2.shape : ", y2.shape)  #(13,)

#1.1
from sklearn.model_selection import train_test_split

x1_predict = x1_predict.reshape(1,3) #(3,) => (1,3)
x2_predict = x2_predict.reshape(1,3) #(3,) => (1,3)

y1 = y1.reshape(10,1) #(1,13) => (13,1)
y2 = y2.reshape(13,1) #(1,13) => (13,1)

x1 = x1.reshape(x1.shape[0],x1.shape[1],1) #(10,3) => (10,3,1)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1) #(13,3) => (13,3,1)

x1_predict = x1_predict.reshape(x1_predict.shape[0],x1_predict.shape[1],1) #(1,3) => (1,3,1)
print("x1_predict.shape : ", x1_predict.shape)  
x2_predict = x2_predict.reshape(x2_predict.shape[0],x2_predict.shape[1],1) #(1,3) => (1,3,1)
print("x2_predict.shape : ", x2_predict.shape)



print("x1.shape : ",x1.shape) #(13,3,1)
print("x2.shape : ",x2.shape) #(13,3,1)

# # shuffle = 랜덤으로 섞음 , random_state (랜덤 난수 표 인덱스) @@@@@@@@@@@@ 필수 @@@@@@@@@@@@
# from sklearn.model_selection import train_test_split
# x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test, = train_test_split(
#     x1, x2, y1, y2, shuffle=False, train_size=0.8 , random_state=66
# )
# x1_train, x1_val, x2_train, x2_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
#     x1_train, x2_train, y1_train, y2_train, train_size = 0.8, test_size=0.2 )

# # MinMaxScaler @@@@@@@@@@@@ 필수 @@@@@@@@@@@@
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x1_train)
# x1_train = scaler.transform(x1_train)
# x1_test = scaler.transform(x1_test)
# x1_val = scaler.transform(x1_val)
# x1_predict = scaler.transform(x1_predict)

# scaler.fit(x2_train)
# x2_train = scaler.transform(x2_train)
# x2_test = scaler.transform(x2_test)
# x2_val = scaler.transform(x2_val)
# x2_predict = scaler.transform(x2_predict)

# x1_train = x1_train.reshape(x1_train.shape[0],x1_train.shape[1],1) #(8,3) => (8,3,1)
# x1_test = x1_test.reshape(x1_test.shape[0],x1_test.shape[1],1) #(3,3) => (3,3,1)
# x1_val = x1_val.reshape(x1_val.shape[0],x1_val.shape[1],1) #(2,3) => (2,3,1)
# x1_predict = x1_predict.reshape(x1_predict.shape[0],x1_predict.shape[1],1) #(1,3) => (1,3,1)

# x_train = [x1_train, x2_train]
# x_test = [x1_test, x2_test]
# x_val = [x1_val, x2_val]
# x_predict = [x1_predict,x2_predict]
# y_train = [y1_train, y2_train]
# y_test = [y1_test, y2_test]

#2. model config
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, LSTM, Input


input1 = Input(shape=(3,1))
Hlayer11 = LSTM(10, activation='relu')(input1)
Hlayer12 = Dense(3, activation='linear')(Hlayer11)
Hlayer13 = Dense(4, activation='linear')(Hlayer12)
Hlayer14 = Dense(4, activation='linear')(Hlayer13)
Hlayer15 = Dense(4, activation='linear')(Hlayer14)

input2 = Input(shape=(3,1))
Hlayer21 = LSTM(10, activation='relu')(input1)
Hlayer22 = Dense(3, activation='linear')(Hlayer21)
Hlayer23 = Dense(4, activation='linear')(Hlayer22)
Hlayer24 = Dense(4, activation='linear')(Hlayer23)
Hlayer25 = Dense(4, activation='linear')(Hlayer24)

#2.3 model Concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate
# from keras.layers import concatenate

merge1 = Concatenate(axis=1)([Hlayer15, Hlayer25])

#2.4 model1 Branch of Output1
output1 = Dense(11)(merge1)
output1 = Dense(10)(output1)
output1 = Dense(1)(output1)

#2.5 model2 Branch of Output2
output2 = Dense(11)(merge1)
output2 = Dense(10)(output2)
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
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='min') 


model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1,
            validation_split=0.2, verbose=1, callbacks=[early_stopping])
            
# model.fit([x1_train,x2_train], y_train, epochs=500, batch_size=1, verbose=1)

# #4. Evaluate, Predict
loss = model.evaluate(x,y)
print("loss : ", loss)

y_pred = model.predict(x_predict)
print(y_pred)
"""
ValueError: Data cardinality is ambiguous:
  x sizes: 10, 13
  y sizes: 10, 13
Please provide data which shares the same first dimension.
"""