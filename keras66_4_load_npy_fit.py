import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPool2D, BatchNormalization

x_train= np.load('../data/image/brain_npy/keras66_train_x.npy')
y_train= np.load('../data/image/brain_npy/keras66_train_y.npy')
x_test= np.load('../data/image/brain_npy/keras66_test_x.npy')
y_test= np.load('../data/image/brain_npy/keras66_test_y.npy')


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5,5),input_shape=(150,150,3),padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(5,5),padding='same'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(
    x_train,y_train, epochs=100, validation_data=(x_test,y_test)
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 시각화 할 것!!

print('acc :', acc[-1])
print('val_acc :', val_acc[:-1])

# acc : 1.0
# val_acc : 0.9583333134651184

import matplotlib.pyplot as plt
# import matplotlib.font_manager as font
plt.figure(figsize=(10,6))          # plot 사이즈
plt.subplot(2,1,1)              # 2행 1열 중 첫번쨰
plt.plot(history.history['loss'], marker='.', c='red', label='loss')
plt.plot(history.history['val_loss'], marker='.', c='blue', label= 'val_loss')
plt.grid() # 격자

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2)              # 2행 2열 중 두번쨰
plt.plot(history.history['acc'], marker='.', c='green', label='mae')
plt.plot(history.history['val_acc'], marker='.', c='yellow', label= 'val_mae')
plt.grid() # 격자

plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()