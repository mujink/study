import numpy as np
x_train = '../data/npy/cifer100_x_train.npy'
x_test = '../data/npy/cifer100_x_test.npy'
y_train = '../data/npy/cifer100_y_train.npy'
y_test = '../data/npy/cifer100_y_test.npy'
x_train = np.load(x_train)
x_test = np.load(x_test)
y_train = np.load(y_train)
y_test = np.load(y_test)

x_train = x_train/255.
x_test = x_test/255.
# (x_test.reshap(10000, 28, 28, 1))

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           
y_train = y_train.reshape(-1,1)                 
y_test = y_test.reshape(-1,1)                 
one.fit(y_train)                          
one.fit(y_test)                          
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()

# plt.imshow(x_train[0])


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 64, kernel_size=(10,10), strides=1,    # kernel_size 자르는 사이즈
     padding= "same", input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=(1,1)))                          # 특성 추출. 
model.add(Activation('relu'))
model.add(Conv2D(filters = 32, kernel_size=(10,10), strides=1,    # kernel_size 자르는 사이즈
     padding= "same"))
model.add(MaxPool2D(pool_size=(1,1)))                          # 특성 추출. 
model.add(Activation('relu'))
model.add(Conv2D(filters = 32, kernel_size=(10,10), strides=1,    # kernel_size 자르는 사이즈
     padding= "same"))
model.add(MaxPool2D(pool_size=(1,1)))                          # 특성 추출. 
model.add(Activation('relu'))
model.add(Flatten())                                            # 1dim
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

print(x_train.shape)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_val, y_val), batch_size= 512)

#4. Evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=512)

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
plt.legend(['train loss', 'val_loss', 'acc','val_acc'])
plt.show()

"""
loss :  1.511629343032837
acc :  0.4738999903202057
R2 : 0.2550318523559663

loss :  1.3921312093734741
acc :  0.5019999742507935
R2 : 0.29649760575850687

loss :  1.4923137426376343
acc :  0.4560999870300293
R2 : 0.24561640973439536
"""
