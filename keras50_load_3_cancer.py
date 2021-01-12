import numpy as np
x = '../data/npy/cancer_x.npy'
y = '../data/npy/cancer_y.npy'
x = np.load(x)
y = np.load(y)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation, Input
from tensorflow.keras.models import Sequential, Model

input1 = Input(shape=(30,))
d1 = Dense(50, activation='sigmoid')(input1)
dh = Dropout(0.1)(d1)
dh = Dense(50, activation='sigmoid')(d1)
dh = Dense(50, activation='sigmoid')(dh)
dh = Dense(20, activation='sigmoid')(dh)
dh = Dense(30, activation='sigmoid')(dh)
dh = Dense(30, activation='sigmoid')(dh)
outputs = Dense(1, activation='sigmoid')(dh)

model = Model(inputs =  input1, outputs = outputs)
model.summary()

#3. Compile, train / binary_corssentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = "../data/modelCheckpoint/k46_MC-6_{epoch:02d}_{val_loss:.4f}.hdf5"  # 가중치 저장 위치
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
cp = ModelCheckpoint(filepath=(modelpath), monitor='val_loss', save_best_only=True, mode='auto')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=300, verbose=1, validation_data=(x_val, y_val), batch_size= 1, callbacks=[early_stopping, cp])
# model.fit(x_train, y_train, epochs=1000)

#4. Evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=114)

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

"""
loss :  0.13297922909259796
acc :  0.9649122953414917
R2 : 0.8640417340173554
"""