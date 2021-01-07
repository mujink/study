import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1,101))

size = 5
def split_x(seq, size):
    aaa=[]                                          # aaa는 list임
    for i in range(len(seq)-size+1):                # for 0에서 seq 변수의 어레이의 길이에서 사이즈의 값을 뺀 만큼 다음을 반복
        subset = seq[i : (i+size)]                  # subset = seq 어레이 [i:(i+size)] 범위의 리스로 초기화함. 
        aaa.append(subset)                          # aaa라는 리스트에 다음 초기화된 [subset]를 값을 맴버로 추가함
    print(type(aaa))
    return np.array(aaa)                            # np.array를 aaa[] 리스트로 초기와 후 반환함.

dataset = split_x(a, size)   #(96,5)
x = dataset[:,:4]            #(96,4)
y = dataset[:,-1]            #(96,)

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


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)      #(60,4) => (60,4,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)          #(20,4) => (20,4,1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)              #(16,4) => (16,4,1)

#2. model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
model = load_model('./model/save_keras35.h5')
model.add(Dense(5,name='asd'))
model.add(Dense(1,name='asf'))
model.summary()

# 3. Compile, train
from tensorflow.keras.callbacks import EarlyStopping
                # mean_squared_error
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto') 

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
            verbose=1, validation_data=(x_val, y_val), callbacks=[early_stopping])

print(hist)
print(hist.history.keys()) #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

print(hist.history['loss'])

# grap
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val_loss', 'acc', 'val_acc'])
plt.show()

#4. Evaluate, Predict
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)


y_Pred = model.predict(x_test[:1,:,:])
print("y_pred : " , y_Pred)
print(y_test.shape)

# from sklearn.metrics import r2_score
# r2_m1 = r2_score(y_test[1], y_Pred)
# print("R2 :", r2_m1 )
# print("y_pred : " , y_Pred)