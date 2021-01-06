# 과제 및 실습
# 전처리, es 등등 다 넣을 것!!!

# 데이터 1~100 / 5개씩 자름
#     x              y
# 1,2,3,4,5          6
# ...
# 95,96,97,98,99,   100

# Predict를 만들 것
#         x_predict         y_predict
# 96,  97,  98,  99,  100 -> 101
# ...
# 100, 101, 102, 103, 104 -> 105
#  예상 Predict는 (101, 102, 103, 104, 105)

import numpy as np
b = np.array(range(96,106))
a = np.array(range(1,101))
size = 6

# Dense 모델 구성하시오.

def split_x(seq, size):
    aaa=[]                                          # aaa는 list임
    for i in range(len(seq)-size+1):                # for 0에서 seq 변수의 어레이의 길이에서 사이즈의 값을 뺀 만큼 다음을 반복
        subset = seq[i : (i+size)]                  # subset = seq 어레이 [i:(i+size)] 범위의 리스로 초기화함. 
        aaa.append(subset)                          # aaa라는 리스트에 다음 초기화된 [subset]를 값을 맴버로 추가함
    # print(type(aaa))
    return np.array(aaa)                            # np.array를 aaa[] 리스트로 초기와 후 반환함.

dataset = np.array(split_x(a, size))

y = dataset[:,-1]                                       # (95,)
x = dataset[:,:5]                                       # (95,5)


predictset = np.array(split_x(b, 6))            # (5,6)
y_pred = predictset[:,-1]                                       # (5,5)
x_pred = predictset[:,:5]                                       # (5,)

print(x_pred[0])
print(x_pred[1])
print(x_pred[2])
print(x_pred[3])
print(x_pred[4])
print(y_pred[0])
print(y_pred[1])
print(y_pred[2])
print(y_pred[3])
print(y_pred[4])


#1.1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler @@@@@@@@@@@@ 필수 @@@@@@@@@@@@

# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# x_pred = scaler.transform(x_pred)



# shuffle = 랜덤으로 섞음 , random_state (랜덤 난수 표 인덱스) @@@@@@@@@@@@ 필수 @@@@@@@@@@@@
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )

print(x_train,'\n')
print(x_test,'\n')
print(x_val,'\n')



scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

print(x_train,'\n')
print(x_test,'\n')
print(x_val,'\n')

print(x_train.shape) #(60,5)
print(x_test.shape)  #(19,5)
print(x_val.shape)   #(16,5)
print(x_pred.shape)  #(5,5)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1) #(95,5) => (95,5,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1) #(95,5) => (95,5,1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1) #(95,5) => (95,5,1)
x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[1],1) #(5,5) => (5,5,1)

print(x_train.shape) #(60,5)
print(x_test.shape)  #(19,5)
print(x_val.shape)   #(16,5)
print(x_pred.shape)  #(5,5)

print(x_train,'\n') #(60,5)
print(x_test,'\n')  #(19,5)
print(x_val,'\n')   #(16,5)
print(x_pred)  #(5,5) 

#2. model config
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Input , LSTM
input1 = Input(shape=(5,1))
Hlayer1 = LSTM(50, activation='relu')(input1)
Hlayer2 = Dense(30, activation='linear')(Hlayer1)
Hlayer3 = Dense(40, activation='linear')(Hlayer2)
Hlayer4 = Dense(40, activation='linear')(Hlayer3)
Hlayer5 = Dense(40, activation='linear')(Hlayer4)
outputs = Dense(1,activation='linear')(Hlayer5)

#2.1 def model
model = Model(inputs =  input1, outputs = outputs)
model.summary()

# 3. Compile, train
from tensorflow.keras.callbacks import EarlyStopping
                # mean_squared_error
early_stopping = EarlyStopping(monitor='loss', patience=300, mode='min') 

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2000, batch_size=1, 
            verbose=1, validation_data=(x_val, y_val), callbacks=[early_stopping])

#4. Evaluate, Predict
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

y_Predict = model.predict(x_pred)
# print("x_test.shape :", x_test.shape) # (2,4,1)

print("x_pred[2] : " , x_pred[2])
print("y_Predict[2] : " , y_Predict[2])
print("x_pred[1] : " , x_pred[1])
print("y_Predict[1] : " , y_Predict[1])

from sklearn.metrics import r2_score
r2_m1 = r2_score(y_pred, y_Predict)
print("R2 :", r2_m1 )

"""
loss :  0.006509724538773298
x_pred[4] :  [[100]
 [101]
 [102]
 [103]
 [104]]
y_Predict[4] :  [105.04974]
x_pred[3] :  [[ 99]
 [100]
 [101]
 [102]
 [103]]
y_Predict[3] :  [104.05038]
R2 : 0.9986988195742015

x_pred[4] :  [[1.05319149]
 [1.05319149]
 [1.05319149]
 [1.05319149]
 [1.05319149]]
y_Predict[4] :  [104.84933]
x_pred[3] :  [[1.04255319]
 [1.04255319]
 [1.04255319]
 [1.04255319]
 [1.04255319]]
y_Predict[3] :  [103.84411]
R2 : 0.9877362135855947

loss :  0.01455533504486084
x_pred[4] :  [[1.05319149]
 [1.05319149]
 [1.05319149]
 [1.05319149]
 [1.05319149]]
y_Predict[4] :  [105.39811]
x_pred[3] :  [[1.04255319]
 [1.04255319]
 [1.04255319]
 [1.04255319]
 [1.04255319]]
y_Predict[3] :  [104.37122]
R2 : 0.9387318691995461

loss :  0.017347391694784164
x_pred[4] :  [[1.05319149]
 [1.05319149]
 [1.05319149]
 [1.05319149]
 [1.05319149]]
y_Predict[4] :  [105.03207]
x_pred[3] :  [[1.04255319]
 [1.04255319]
 [1.04255319]
 [1.04255319]
 [1.04255319]]
y_Predict[3] :  [104.00509]
R2 : 0.9992539058905094

loss :  0.016265977174043655
x_pred[2] :  [[1.03191489]
 [1.03191489]
 [1.03191489]
 [1.03191489]
 [1.03191489]]
y_Predict[2] :  [102.95593]
x_pred[1] :  [[1.0212766]
 [1.0212766]
 [1.0212766]
 [1.0212766]
 [1.0212766]]
y_Predict[1] :  [101.93789]
R2 : 0.9987619576742872

"""

