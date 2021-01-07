# keras23_LSTM3_scale 을 함수형으로 코딩

#1. data

import numpy as np

x= np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],[20,30,40],
            [30,40,50],[40,50,60]])
y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x.shape : ", x.shape)  #(13,3)
print("y.shape : ", y.shape)  #(13,)

# print("x.shape : ", x.shape)  #(13,3,1)



#1.1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler @@@@@@@@@@@@ 필수 @@@@@@@@@@@@

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x = x.reshape(13,3,1)

# x = x.reshape(13,3,1)
# print("x.shape : ", x.shape)  #(13,3,1)

# shuffle = 랜덤으로 섞음 , random_state (랜덤 난수 표 인덱스) @@@@@@@@@@@@ 필수 @@@@@@@@@@@@
x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )


#2. model config
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Input , LSTM
input1 = Input(shape=(3,1))
Hlayer1 = LSTM(5, activation='relu')(input1)
Hlayer2 = Dense(3, activation='linear')(Hlayer1)
Hlayer3 = Dense(4, activation='linear')(Hlayer2)
Hlayer4 = Dense(4, activation='linear')(Hlayer3)
Hlayer5 = Dense(4, activation='linear')(Hlayer4)
outputs = Dense(1,activation='linear')(Hlayer5)
model = Model(inputs =  input1, outputs = outputs)
model.summary()

#3. Compile, train
from tensorflow.keras.callbacks import EarlyStopping
                # mean_squared_error
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='min') 

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1, verbose=1)

#4. Evaluate, Predict
loss = model.evaluate(x,y)
print("loss : ", loss)


y_Pred = model.predict(x_test)
print(x_test.shape) # (3,3,1)

# 
from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_Pred)
print("y_pred : " , y_Pred)

print("R2 :", r2_m1 )
# print("y_pred : " , y_Pred)



"""
loss :  4.8170294761657715
[[7.57495]]

loss :  0.444517582654953
[[8.687467]]

oss :  0.012537196278572083
[[7.9974117]]

loss :  2.183164358139038
[[8.483552]]


All layer activation='linear'@@ => 더 잘나옴
loss :  0.0024970872327685356
[[7.9833612]]

loss :  0.13988704979419708
[[7.5443316]]

loss :  0.0075500477105379105
[[7.9712753]]

loss :  0.18634793162345886
[[7.909652]]

loss :  7.346150875091553
[[7.700137]]

loss :  0.004054872319102287
[[8.004452]]

x_pred = np.array([50,60,70])
loss :  0.11998085677623749
[[77.331566]]

loss :  0.004336968995630741
[[80.646]]

loss :  0.004388676956295967
[[82.942894]]

patience=100
loss :  0.023663656786084175
[[80.41219]]

loss :  0.0871659442782402
[[86.59359]]

loss :  0.024140801280736923
[[81.300766]]


loss :  0.013255717232823372
y_pred :  [[10.001777]
 [ 5.035692]
 [59.70413 ]]
R2 : 0.9999519907166478
"""