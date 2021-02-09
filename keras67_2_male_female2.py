#  실습
#  imageDataGenerator fit 사용

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPool2D, BatchNormalization, AveragePooling2D


train_datagen = ImageDataGenerator(
    # 받게될 이미지를 아래 내용기준으로 변환한다
    # 스케일 한다
    rescale=1./255,
    # 변환파라미터
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=0.1,
    zoom_range=0.4,
    validation_split= 0.5,
    # 사진을 휘거나 늘어뜨림
    # shear_range=0.5,
    # 이동 후 빈자리를 nearest 모드를 한다.
    fill_mode='nearest'
)

# flow 또는 flow_from_directory(폴더명을 카테고리로 지정한 경우 폴더명이 Y라벨이됨)
# 이미지 플로트 하는 걸 과제로 낼거임

# test_datagen에 대해서 프롤우 프럼 디렉토리를 다음과 같이한다.

xy_train = train_datagen.flow_from_directory(
    # 첫번 째 인자는 디렉토리 폴더를 받음
    '../data/image/data2', # (160,256,256,3)
    # 폴더 안에 있는 전체 이미지 데이터를 사이즈를 바꿈
    color_mode ='grayscale',
    target_size=(100,100), # (160,150,150,3)
    batch_size=8, # (전체 데이터/batch_size, batch_size,150,150,3 5차원 배열)
    class_mode='binary',
    subset="training",
)

xy_test = train_datagen.flow_from_directory(
    # 첫번 째 인자는 디렉토리 폴더를 받음
    '../data/image/data2', # (160,256,256,3)
    # 폴더 안에 있는 전체 이미지 데이터를 사이즈를 바꿈
    color_mode ='grayscale',
    target_size=(100,100), # (160,150,150,3)
    batch_size=8, # (전체 데이터/batch_size, batch_size,150,150,3 5차원 배열)

    class_mode='binary',
    subset="validation",
)

np.save('../data/image/data2_npy/keras67_train_x.npy', arr=xy_train[0][0])
np.save('../data/image/data2_npy/keras67_train_y.npy', arr=xy_train[0][1])
np.save('../data/image/data2_npy/keras67_test_x.npy', arr=xy_test[0][0])
np.save('../data/image/data2_npy/keras67_test_y.npy', arr=xy_test[0][1])

x_train= np.load('../data/image/data2_npy/keras67_train_x.npy')
y_train= np.load('../data/image/data2_npy/keras67_train_y.npy')
x_test= np.load('../data/image/data2_npy/keras67_test_x.npy')
y_test= np.load('../data/image/data2_npy/keras67_test_y.npy')

model = Sequential()
model.add(Conv2D(filters=3, kernel_size=(8,8),input_shape=(100,100,1),padding='same',activation='relu'))
model.add(AveragePooling2D((2,2),(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides = (2,2),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides = (2,2),padding='same'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# 핏제너레이터는 x,y를 나누지 않고 넣음

history = model.fit(
    x_train,y_train, epochs=100, validation_data=(x_test,y_test)
)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
print('acc :', acc[-1])
print('val_acc :', val_acc[-1])

loss1, acc1 = model.evaluate(x_test, y_test)
print("loss :", loss1)
print("acc :", acc1)

del xy_train
del xy_test

# 시각화 할 것!!

# acc : 0.8999999761581421
# 0.949999988079071

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