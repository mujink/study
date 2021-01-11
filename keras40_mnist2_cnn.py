#  인공지능게의  helloe world => Mnist!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)             # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)               # (10000, 28, 28) (10000,)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1).astype('float32')/255.
# (x_test.reshap(10000, 28, 28, 1))

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           
y_train = y_train.reshape(-1,1)                 
y_test = y_test.reshape(-1,1)                 
one.fit(y_train)                          
one.fit(y_test)                          
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()


# print(y_train.shape)
# print(y_train.shape)

# print(y_test.shape)
# print(x_train.shape)
# print(x_test.shape)
print("=============")




from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 10, kernel_size=(10,10), strides=1,    # kernel_size 자르는 사이즈
     padding= "same", input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(1,1)))                          # 특성 추출. 
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), padding='valid'))
model.add(MaxPool2D(pool_size=(2,2)))                          # 특성 추출. 
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Flatten())                                            # 1dim
model.add(Dense(10, activation='relu'))
# model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()


#3. Compile, train / binary_corssentropy

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, epochs=1000, verbose=1, callbacks=[early_stopping], batch_size= 60000)
hist = model.fit(x_train, y_train, epochs=5, verbose=1, batch_size= 30)
# model.fit(x_train, y_train, epochs=1000)

#4. Evaluate, predict
loss, accuracy = model.evaluate(x_test, y_test, batch_size=3)

print("loss : ", loss)
print("acc : ", accuracy)



y_pred = model.predict(x_test[:10])
print(y_pred.argmax(axis=1))
print(y_test.argmax(axis=1))

print("    예상 출력     /    실제  값     ")
# print("    ",y_pred[:10],"     /    ", y_test[:10])

# fit.. hist

print(hist)
print(hist.history.keys()) #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])
plt.title('cnn mnist')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'acc'])
plt.show()

"""
loss :  0.06191143020987511
acc :  0.9811000227928162
"""
