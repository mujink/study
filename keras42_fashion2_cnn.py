수정필요
# cnn으로 구성
# 2차원을 4차원으로 늘여서 하시오/

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)             # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)               # (10000, 28, 28) (10000,)

# print(x_train[0])
# print(x_train[0].shape)
print("y_train[0] :",y_train[0])


print(x_train.min(), x_train.max())
# (x_test.reshap(10000, 28, 28, 1))




from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1).astype('float32')/255.
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1).astype('float32')/255.


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           
y_train = y_train.reshape(-1,1)                 
y_test = y_test.reshape(-1,1)                 
y_val = y_val.reshape(-1,1)                 
one.fit(y_train)                          
one.fit(y_test)                          
one.fit(y_val)                          
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
y_val = one.transform(y_val).toarray()

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 5, kernel_size=(2,2),    # kernel_size 자르는 사이즈
     padding= "same", input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))                          # 특성 추출. 
model.add(Activation('relu'))
model.add(Flatten())                                            # 1dim
model.add(Dense(100, activation='linear'))
model.add(Dense(60, activation='linear'))
model.add(Dense(30, activation='linear'))
# model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

print(x_train.shape)
#3. Compile, train / binary_corssentropy

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_val, y_val), batch_size= 3000)
# model.fit(x_train, y_train, epochs=1000)

#4. Evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=3)

print("loss : ", loss)
print("acc : ", mae)

y_predict = model.predict(x_test)


from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_predict)

print("R2 :", r2_m1 )


# fit.. hist

print(hist)
print(hist.history.keys()) #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])
plt.title('cnn fashion_mnist')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'acc'])
plt.show()

"""
loss :  0.022198529914021492
acc :  0.8422999978065491
R2 : 0.7533497232231358

loss :  0.015136634930968285
acc :  0.8974999785423279
R2 : 0.8318149894009566

loss :  0.3407743573188782
acc :  0.8847000002861023
R2 : 0.813372813297287

loss :  0.38622480630874634
acc :  0.86080002784729
R2 : 0.7788237894388813

loss :  0.34943294525146484
acc :  0.8755999803543091
R2 : 0.7999088993919219

loss :  0.3655332326889038
acc :  0.871999979019165
R2 : 0.7935510532388719
"""