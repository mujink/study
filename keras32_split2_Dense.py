import numpy as np

a = np.array(range(1,11))
size = 5

# Dense 모델 구성하시오.

def split_x(seq, size):
    aaa=[]                                          # aaa는 list임
    for i in range(len(seq)-size+1):                # for 0에서 seq 변수의 어레이의 길이에서 사이즈의 값을 뺀 만큼 다음을 반복
        subset = seq[i : (i+size)]                  # subset = seq 어레이 [i:(i+size)] 범위의 리스로 초기화함. 
        aaa.append(subset)                          # aaa라는 리스트에 다음 초기화된 [subset]를 값을 맴버로 추가함
    print(type(aaa))
    return np.array(aaa)                            # np.array를 aaa[] 리스트로 초기와 후 반환함.

dataset = np.array(split_x(a, size))
y = dataset[:,-1]                                       # (6,)
x = dataset[:,:4]                                       # (6,4)
print(y)
print(x)


#1.1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler @@@@@@@@@@@@ 필수 @@@@@@@@@@@@
"""
 안쓰는게 잘나옴
"""
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)

# x = x.reshape(6,4,1)

# shuffle = 랜덤으로 섞음 , random_state (랜덤 난수 표 인덱스) @@@@@@@@@@@@ 필수 @@@@@@@@@@@@
x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,train_size = 0.8, test_size=0.2)

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)


#2. model config
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Input , LSTM
input1 = Input(shape=(4,))
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


y_Pred = model.predict(x_test)
# print("x_test.shape :", x_test.shape) # (2,4)

print("x_test[0] : " , x_test[0])
print("y_pred[0] : " , y_Pred[0])
print("x_test[1] : " , x_test[1])
print("y_pred[1] : " , y_Pred[1])

from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_Pred)
print("R2 :", r2_m1 )

# from sklearn.metrics import r2_score
# r2_m1 = r2_score(y_test, y_Pred)
# print("R2 :", r2_m1 )
# # print("y_pred : " , y_Pred)

"""
loss :  0.0012278197100386024
x_test[0] :  [1 2 3 4]
y_pred[0] :  [4.9527965]
x_test[1] :  [2 3 4 5]
y_pred[1] :  [5.9849195]
R2 : 0.9950888114085501
"""