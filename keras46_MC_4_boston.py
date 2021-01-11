import numpy as np
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(50, input_dim=13, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

#3. Compile, train / binary_corssentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = "./modelCheckpoint/k46_MC-4_{epoch:02d}_{val_loss:.4f}.hdf5"  # 가중치 저장 위치
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
cp = ModelCheckpoint(filepath=(modelpath), monitor='val_loss', save_best_only=True, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=300, verbose=1, validation_data=(x_val, y_val), batch_size= 32, callbacks=[early_stopping, cp])
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
plt.plot(hist.history['mae'], marker='.', c='green', label='mae')
plt.plot(hist.history['val_mae'], marker='.', c='yellow', label= 'val_mae')
plt.grid() # 격자

plt.title('mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

"""
loss :  18.301071166992188
acc :  0.0
R2 : 0.8148178532131564

loss :  18.589366912841797
mae :  3.4682559967041016
R2 : 0.811900682586622
"""