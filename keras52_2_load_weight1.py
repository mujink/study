
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
from tensorflow.keras.models import Sequential, load_model

model2 = Sequential()
model2.add(Conv2D(filters = 10, kernel_size=(10,10), strides=1,    # kernel_size 자르는 사이즈
     padding= "same", input_shape=(28,28,1)))
model2.add(MaxPool2D(pool_size=(1,1)))                          # 특성 추출. 
model2.add(Activation('relu'))
model2.add(Dropout(0.2))
model2.add(Conv2D(10, (2,2), padding='valid'))
model2.add(MaxPool2D(pool_size=(2,2)))                          # 특성 추출. 
model2.add(Activation('relu'))
model2.add(Flatten())                                            # 1dim
model2.add(Dense(20, activation='relu'))
model2.add(Dense(10, activation='softmax'))

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# model load 모델 및 가중치 로드 컴파일 및 모델 필요없음.
model1 = load_model('../data/h5/k52_1_model2.h5') 
# model.save('../data/h5/k52_1_model2.h5')

#4. Evaluate, predict
model1_loss, model1_mae = model1.evaluate(x_test, y_test, batch_size=3)
print("load_model2_loss : ", model1_loss)
print("load_model2_acc : ", model1_mae)

# 3334/3334 [==============================] - 3s 1ms/step - loss: 0.0407 - acc: 0.9877
# load_model2_loss :  0.04067131504416466
# load_model2_acc :  0.9876999855041504

# weights load 웨이트 값만 로드시 모델 및 컴파일이 필요함.
model2.load_weights('../data/h5/k52_1_weight.h5')
# model.save_weights('../data/h5/k52_1_weight.h5')

model2_loss, model2_mae = model2.evaluate(x_test, y_test, batch_size=3)
print("load_weights_loss : ", model2_loss)
print("load_weights_acc : ", model2_mae)

# 3334/3334 [==============================] - 3s 1ms/step - loss: 0.0407 - acc: 0.9877
# load_weights_loss :  0.04067131504416466
# load_weights_acc :  0.9876999855041504