
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)             # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)               # (10000, 28, 28) (10000,)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1).astype('float32')/255.

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           
y_train = y_train.reshape(-1,1)                 
y_test = y_test.reshape(-1,1)                 
one.fit(y_train)                          
one.fit(y_test)                          
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()

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
model.add(Flatten())                                            # 1dim
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 모델만 저장할 경우 모델세이브 위치.
model.save('../data/h5/k52_1_model1.h5') 

#3. Compile, train / binary_corssentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = "../data/modelCheckpoint/k52_1_mnist_MCK_{epoch:02d}_{val_loss:.4f}.hdf5"  # 가중치 저장 위치 (.hdf5)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
cp = ModelCheckpoint(filepath=(modelpath), monitor='val_loss', save_best_only=True, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, verbose=1, validation_split=0.2, batch_size= 32, callbacks=[early_stopping, cp])
# model.fit(x_train, y_train, epochs=1000)


# 가중치와 모델을 함께 저장할 경우 모델세이브 위치.
model.save('../data/h5/k52_1_model2.h5')
# 핏 후 저장하면 가중치도 같이 저장됨.

# 가중치 저장 (h5)
model.save_weights('../data/h5/k52_1_weight.h5')

#4. Evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=3)

print("loss : ", loss)
print("acc : ", mae)
