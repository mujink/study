수정필요

# cnn으로 구성
# 2차원을 4차원으로 늘여서 하시오

#  인공지능게의  helloe world => Mnist!!

import numpy as np
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape) #(506, 13)
print(y.shape) #(506,)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=33
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,  test_size=0.2
)

# x의 값들을 0~1 사이 값으로 줄임 => 가중치를 낮추어 연산 속도가 빨라짐.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1,1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1,1)
# (x_test.reshap(506, 13, 1, 1))


from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 20, kernel_size=(1,1), input_shape=(x_train.shape[1],1,1)))

model.add(Flatten())                                            # 1dim
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
# model.add(Dense(100))
model.add(Dense(1, activation='linear'))

model.summary()

print(x_train.shape)
#3. Compile, train / binary_corssentropy

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=5000, verbose=1, batch_size= 10, validation_data=(x_val, y_val))
# model.fit(x_train, y_train, epochs=1000)

#4. Evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=3)

print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)


from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_predict)

print("R2 :", r2_m1 )


# fit.. hist

print(hist)
print(hist.history.keys()) #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['mae'])
plt.title('cnn boston')
plt.ylabel('loss, mae')
plt.xlabel('epoch')
plt.legend(['train loss', 'mae'])
plt.show()

"""
loss :  21.874467849731445
mae :  3.551494598388672
R2 : 0.6946376654874854
"""
