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

# LSTM 과 결과 비교!!
# Dense 모델 구성하시오.
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
    print(type(aaa))
    return np.array(aaa)                            # np.array를 aaa[] 리스트로 초기와 후 반환함.

dataset = np.array(split_x(a, size))
y = dataset[:,-1]                                       # (95,)
x = dataset[:,:5]                                       # (95,5)
predictset = np.array(split_x(b, 6))            # (5,6)
y_pred = predictset[:,-1]                                       # (5,5)
x_pred = predictset[:,:5]                                       # (5,)


#1.1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# shuffle = 랜덤으로 섞음 , random_state (랜덤 난수 표 인덱스) @@@@@@@@@@@@ 필수 @@@@@@@@@@@@
x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,train_size = 0.8, test_size=0.2)

# MinMaxScaler @@@@@@@@@@@@ 필수 @@@@@@@@@@@@

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)


#2. model config
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Input , LSTM
input1 = Input(shape=(5,))
Hlayer1 = Dense(50, activation='relu')(input1)
Hlayer2 = Dense(30, activation='relu')(Hlayer1)
Hlayer3 = Dense(40, activation='relu')(Hlayer2)
Hlayer4 = Dense(40, activation='relu')(Hlayer3)
Hlayer5 = Dense(40, activation='relu')(Hlayer4)
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
# print("x_test.shape :", x_test.shape) # (2,4)

print("x_pred[4] : " , x_pred[4])
print("y_pred[4] : " , y_Predict[4])
print("x_pred[3] : " , x_pred[3])
print("y_pred[3] : " , y_Predict[3])

from sklearn.metrics import r2_score
r2_m1 = r2_score(y_pred, y_Predict)
print("R2 :", r2_m1 )

"""
loss :  1.123240849665308e-06
x_test[0] :  [ 96  97  98  99 100]
y_pred[0] :  [100.99957]
x_test[1] :  [ 97  98  99 100 101]
y_pred[1] :  [101.99956]
R2 : 0.9999998859479092

x_test[4] :  [100 101 102 103 104]
y_pred[4] :  [105.1208]
x_test[3] :  [ 99 100 101 102 103]
y_pred[3] :  [104.11972]
R2 : 0.9929621697636322

loss :  1.7457574358559214e-05
x_pred[4] :  [1.05434783 1.05434783 1.05434783 1.05434783 1.05434783]
y_pred[4] :  [105.00805]
x_pred[3] :  [1.04347826 1.04347826 1.04347826 1.04347826 1.04347826]
y_pred[3] :  [104.007835]
R2 : 0.9999707961163949
"""