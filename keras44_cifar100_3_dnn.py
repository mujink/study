수정필요

from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar100.load_data()


# plt.imshow(x_train[0])
# plt.show()


# plt.imshow(x_train[0])
# plt.show()

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*3).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*3).astype('float32')/255.
# (x_test.reshap(10000, 28, 28, 1))

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           
y_train = y_train.reshape(-1,1)                 
y_test = y_test.reshape(-1,1)                 
one.fit(y_train)                          
one.fit(y_test)                          
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(150, activation='softmax', input_shape=(3072,)))
model.add(Dense(100, activation='softmax'))
model.add(Dense(100, activation='softmax'))

model.summary()

#3. Compile, train / binary_corssentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=50, verbose=1, validation_data=(x_val, y_val), batch_size= 100)

#4. Evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=120)

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
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('cnn fashion_mnist')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss','val_loss', 'acc','val_acc'])
plt.show()

"""
loss :  3.3592753410339355
acc :  0.20569999516010284
R2 : 0.0926101207157381
"""