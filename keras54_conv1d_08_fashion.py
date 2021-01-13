# 실습
# conv1d로 코딩하시오

import numpy as np
x_train = '../data/npy/tahion_x_train.npy'
x_test = '../data/npy/tahion_x_test.npy'
y_train = '../data/npy/tahion_y_train.npy'
y_test = '../data/npy/tahion_y_test.npy'
x_train = np.load(x_train)
x_test = np.load(x_test)
y_train = np.load(y_train)
y_test = np.load(y_test)


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2]).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2]).astype('float32')/255.
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2]).astype('float32')/255.
# (x_test.reshap(10000, 28, 28, 1))

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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation, Conv1D, Flatten,MaxPool1D


# model = Sequential()
# model.add(Conv1D(filters = 10, kernel_size=(10), strides=1,    # kernel_size 자르는 사이즈
#      padding= "same", input_shape=(28,28)))
# # model.add(MaxPool1D(pool_size=(2)))                          # 특성 추출. 
# # model.add(Activation('relu'))
# # model.add(Dropout(0.2))
# model.add(Flatten())                                            # 1dim
# model.add(Dense(50, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(15, activation='relu'))
# # model.add(Dense(100))
# model.add(Dense(10, activation='softmax'))
# model.summary()

model = Sequential()
model.add(Conv1D(10,2 ,input_shape=(x_train.shape[1],x_train.shape[2]) ,padding='same'))
model.add(MaxPool1D(pool_size=(2)))                          # 특성 추출. 
model.add(Activation('relu'))
model.add(Flatten())                                            # 1dim
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

#3. Compile, train / binary_corssentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = "../data/modelCheckpoint/k46_MC-1_{epoch:02d}_{val_loss:.4f}.hdf5"  # 가중치 저장 위치
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
# cp = ModelCheckpoint(filepath=(modelpath), monitor='val_loss', save_best_only=True, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_val, y_val), batch_size= 16, callbacks=[early_stopping])#, cp])
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

import matplotlib.pyplot as plt
import matplotlib.font_manager as font
plt.figure(figsize=(10,6))          # plot 사이즈
plt.subplot(2,1,1)              # 2행 1열 중 첫번쨰
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label= 'val_loss')
plt.grid() # 격자

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2)              # 2행 2열 중 두번쨰
plt.plot(hist.history['acc'], marker='.', c='green', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='yellow', label= 'val_acc')
plt.grid() # 격자

plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

"""Conv2D
loss :  0.2840123176574707
acc :  0.9013000130653381
R2 : 0.839886782703406
"""
"""c1d 
loss :  0.37659716606140137
acc :  0.8648999929428101
R2 : 0.7839271387289155

loss :  0.3552562892436981
acc :  0.8698999881744385
R2 : 0.7942673705709081

loss :  0.36747655272483826
acc :  0.8726000189781189
R2 : 0.7979000574082934

loss :  0.35414230823516846
acc :  0.8770999908447266
R2 : 0.8016937279705741
"""