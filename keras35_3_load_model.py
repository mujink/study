
#1. data
import numpy as np
a = np.array(range(1,11))

#1.1 preprocess data

size = 5
def split_x(seq, size):
    aaa=[]                                          # aaa는 list임
    for i in range(len(seq)-size+1):                # for 0에서 seq 변수의 어레이의 길이에서 사이즈의 값을 뺀 만큼 다음을 반복
        subset = seq[i : (i+size)]                  # subset = seq 어레이 [i:(i+size)] 범위의 리스로 초기화함. 
        aaa.append(subset)                          # aaa라는 리스트에 다음 초기화된 [subset]를 값을 맴버로 추가함
    print(type(aaa))
    return np.array(aaa)                            # np.array를 aaa[] 리스트로 초기와 후 반환함.

dataset = split_x(a, size)
x = dataset[:,:4]
y = dataset[:,4:5]


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler @@@@@@@@@@@@ 필수 @@@@@@@@@@@@





# shuffle = 랜덤으로 섞음 , random_state (랜덤 난수 표 인덱스) @@@@@@@@@@@@ 필수 @@@@@@@@@@@@
x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,train_size = 0.8, test_size=0.2, shuffle=True, random_state=66 )

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

#2. model
from tensorflow.keras.models import load_model
model = load_model('./Data/h5/save_keras35.h5')
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
# print("x_test.shape :", x_test.shape) # (2,4,1)
print("x_test[0] : " , x_test[0])
print("y_pred[0] : " , y_Pred[0])
print("x_test[1] : " , x_test[1])
print("y_pred[1] : " , y_Pred[1])

# print(y_Pred)
# print(y_test)
# print(y)
# print(x)

from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_Pred)
print("R2 :", r2_m1 )
print("y_pred : " , y_Pred)
"""
loss :  0.0016826787032186985
x_test[0] :  [[-1.]
 [-1.]
 [-1.]
 [-1.]]
y_pred[0] :  [5.057444]
x_test[1] :  [[-0.5]
 [-0.5]
 [-0.5]
 [-0.5]]
y_pred[1] :  [6.0080953]
R2 : 0.9932692851461979
y_pred :  [[5.057444 ]
[6.0080953]]
"""