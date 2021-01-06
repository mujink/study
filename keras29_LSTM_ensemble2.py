#  predict 85근사치 출력
# 2개의 모델을 하나는 LSTM, 하나는 Dense로 앙상블 구현
#1. data

import numpy as np
from numpy import array
# x = np.array([range(3), range(3), range(3)])

x1= np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
x2= np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]])
y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = array([55,65,75])  #(3,)
x2_predict = array([65,75,85])  #(3,)

print("x1.shape : ", x1.shape)  #(13,3)
print("x2.shape : ", x2.shape)  #(13,3)
print("y.shape : ", y.shape)  #(13,)

#1.1
from sklearn.model_selection import train_test_split

# MinMaxScaler @@@@@@@@@@@@ 필수 @@@@@@@@@@@@


x1_predict = x1_predict.reshape(1,3) #(3,) => (1,3)
x2_predict = x2_predict.reshape(1,3) #(3,) => (1,3)
x1 = x1.reshape(x1.shape[0],x1.shape[1],1) #(13,3) => (13,3,1)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1) #(13,3) => (13,3,1)

y = y.reshape(13,1) #(1,13) => (13,1)

# x2 = x1.reshape(x2.shape[0],x2.shape[1],1) #(13,3) => (13,3,1)


# x2_predict = x2_predict.reshape(x2_predict.shape[0],x1_predict.shape[1],1) #(1,3) => (1,3,1)
print("x2_predict.shape : ", x2_predict.shape)

print("x1.shape : ",x1.shape) #(13,3,1)
print("x2.shape : ",x2.shape) #(13,3,1)

# shuffle = 랜덤으로 섞음 , random_state (랜덤 난수 표 인덱스) @@@@@@@@@@@@ 필수 @@@@@@@@@@@@
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test, = train_test_split(
    x1, x2, y, shuffle=False, train_size=0.8 , random_state=66
)
x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(
    x1_train, x2_train, y_train, train_size = 0.8, test_size=0.2 )

print("x1_train.shape :",x1_train.shape)
print("x1_test.shape :",x1_test.shape)
print("x1_val.shape :",x1_val.shape)
print("x1_predict.shape :",x1_predict.shape)

print("x2_train.shape :",x2_train.shape)
print("x2_test.shape :",x2_test.shape)
print("x2_val.shape :",x2_val.shape)
print("x2_predict.shape :",x2_predict.shape)




print("x1_train.shape :",x1_train.shape)
print("x1_test.shape :",x1_test.shape)
print("x1_val.shape :",x1_val.shape)
print("x1_predict.shape :",x1_predict.shape)

print("x2_train.shape :",x2_train.shape)
print("x2_test.shape :",x2_test.shape)
print("x2_val.shape :",x2_val.shape)
print("x2_predict.shape :",x2_predict.shape)

# 2. model config
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, LSTM, Input

input1 = Input(shape=(3,1))
Hlayer11 = LSTM(10, activation='relu')(input1)
Hlayer12 = Dense(3, activation='relu')(Hlayer11)
Hlayer13 = Dense(4, activation='relu')(Hlayer12)
Hlayer14 = Dense(4, activation='relu')(Hlayer13)
Hlayer15 = Dense(4, activation='relu')(Hlayer14)

input2 = Input(shape=(3,))
Hlayer21 = Dense(10, activation='linear')(input2)
Hlayer22 = Dense(3, activation='linear')(Hlayer21)
Hlayer23 = Dense(4, activation='linear')(Hlayer22)
Hlayer24 = Dense(4, activation='linear')(Hlayer23)
Hlayer25 = Dense(4, activation='linear')(Hlayer24)

#2.3 model Concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate
# from keras.layers import concatenate

merge1 = Concatenate(axis=1)([Hlayer15, Hlayer25])
middle1 = Dense(3)(merge1)
middle1 = Dense(5)(middle1)
middle1 = Dense(9)(middle1)

#2.4 model1 Branch of Output1
output1 = Dense(11)(middle1)
output1 = Dense(10)(output1)
output1 = Dense(1)(output1)

#2.5 def model 
inputs = [input1, input2]
model = Model(inputs =  inputs, outputs = output1)
model.summary()


# 3. Compile, train

# from tensorflow.keras.callbacks import EarlyStopping
#                 # mean_squared_error
# early_stopping = EarlyStopping(monitor='loss', patience=30, mode='min') 

model.compile(loss='mse', optimizer='adam')
# model.fit([x1_train,x2_train], y_train, epochs=500, batch_size=1,
            # validation_data=([x1_val,x2_val]), verbose=1, callbacks=[early_stopping])
model.fit([x1_train,x2_train], y_train, epochs=500, batch_size=1,
          validation_data=([x1_val,x2_val]), verbose=1)

# #4. Evaluate, Predict
loss = model.evaluate([x1_test,x2_test],y_test)
print("loss : ", loss)

y_pred = model.predict([x1_predict,x2_predict])
print("x1_predict.shape : ", x1_predict.shape)
print("x2_predict.shape : ", x2_predict.shape)
print("y_pred.shape : ", y_pred.shape)
print("y_pred : ", y_pred)

# y_predict = np.array([[85]])
# print("y_predict.shape : ", y_predict.shape)
# print("y_predict : ", y_predict)


# from sklearn.metrics import r2_score
# r2_m1 = r2_score(y_predict,y_pred)
# print("R2 :", r2_m1 )



"""
loss :  106.3257827758789
[[100.61249]]

loss :  271.7812194824219
[[88.50581]]

loss :  258.2997131347656
[[87.70701]]

loss :  126.46092987060547
[[109.52122]]

loss :  552.4878540039062
[[68.73371]]
"""