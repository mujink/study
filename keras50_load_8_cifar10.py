import numpy as np
x_train = '../data/npy/cifer10_x_train.npy'
x_test = '../data/npy/cifer10_x_test.npy'
y_train = '../data/npy/cifer10_y_train.npy'
y_test = '../data/npy/cifer10_y_test.npy'
x_train = np.load(x_train)
x_test = np.load(x_test)
y_train = np.load(y_train)
y_test = np.load(y_test)

print(x_train.shape)
print(x_test.shape)

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
model.add(Conv2D(filters = 10, kernel_size=(10,10), strides=1,    # kernel_size 자르는 사이즈
     padding= "same", input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=(1,1)))                          # 특성 추출. 
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), padding='valid'))
model.add(MaxPool2D(pool_size=(2,2)))                          # 특성 추출. 
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Flatten())                                            # 1dim
model.add(Dense(20, activation='relu'))
# model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()


#3. Compile, train / binary_corssentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = "./data/modelCheckpoint/k46_MC-2_{epoch:02d}_{val_loss:.4f}.hdf5"  # 가중치 저장 위치
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_val, y_val), batch_size= 256, callbacks=[early_stopping])
# model.fit(x_train, y_train, epochs=1000)

#4. Evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=256)

print("loss : ", loss)
print("acc : ", mae)

y_predict = model.predict(x_test)


from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_predict)

print("R2 :", r2_m1 )


# fit.. hist

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
loss :  1.4977484941482544
acc :  0.508400022983551
R2 : 0.287518364391902
"""