#  실습
#  imageDataGenerator fit_gerator 사용
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
    zoom_range=0.2,
    validation_split= 0.2,
    # 사진을 휘거나 늘어뜨림
    # shear_range=0.5,
    # 이동 후 빈자리를 nearest 모드를 한다.
    fill_mode='nearest'
)

# flow 또는 flow_from_directory(폴더명을 카테고리로 지정한 경우 폴더명이 Y라벨이됨)
# 이미지 플로트 하는 걸 과제로 낼거임

# test_datagen에 대해서 프롤우 프럼 디렉토리를 다음과 같이한다.
def ImageDatatrain ():
    dataset = train_datagen.flow_from_directory(

        # '../data/image/data2/fm', 
        '../data/image/data2_generator/', 

        color_mode ='grayscale',
        target_size=(100,100),
        batch_size=75,
        class_mode='binary',
        subset="training",
        # save_to_dir='../data/image/data2_generator/fm'
    )
    return dataset

def ImageDataval ():
    dataset = train_datagen.flow_from_directory(

        # '../data/image/data2/m',
        '../data/image/data2_generator/', 

        color_mode ='grayscale',
        target_size=(100,100),
        batch_size=75,

        class_mode='binary',
        subset="validation",
        # save_to_dir='../data/image/data2_generator/m'
    )
    return dataset

xy_train =[]
xy_val = []
xy_train = ImageDatatrain()
xy_val = ImageDataval()


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
history = model.fit_generator(

    xy_train, steps_per_epoch=10, epochs=10,validation_data=xy_val,
    validation_steps=4
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
print('acc :', acc[-1])
print('val_acc :', val_acc[-1])

xy_test = ImageDataval()
x_test = xy_test[0][0]
y_test = xy_test[0][1]

loss1, acc1 = model.evaluate(x_test, y_test)
print("loss :", loss1)
print("acc :", acc1)

del xy_val
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

# loss : 39.598358154296875
# acc : 0.625 
