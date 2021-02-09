import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPool2D, BatchNormalization


train_datagen = ImageDataGenerator(
    # 받게될 이미지를 아래 내용기준으로 변환한다
    # 스케일 한다
    rescale=1./255,
    # 변환파라미터
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    # zoom_range=1.5,
    # 사진을 휘거나 늘어뜨림
    # shear_range=0.5,
    # 이동 후 빈자리를 nearest 모드를 한다.
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory(폴더명을 카테고리로 지정한 경우 폴더명이 Y라벨이됨)
# 이미지 플로트 하는 걸 과제로 낼거임

# test_datagen에 대해서 프롤우 프럼 디렉토리를 다음과 같이한다.
xy_train = train_datagen.flow_from_directory(
    # 첫번 째 인자는 디렉토리 폴더를 받음
    '../data/image/brain/train', # (160,256,256,3)
    # 폴더 안에 있는 전체 이미지 데이터를 사이즈를 바꿈
    target_size=(150,150), # (160,150,150,3)
    batch_size=10, # (전체 데이터/batch_size, batch_size,150,150,3 5차원 배열)
                  # (만약 베치사이즈가 전체 데이터 사이즈보다 크면 4차원 배열 생성됨)
    # 폴더 안에 있는 이미지에 대해서 라벨값을 지정하는데
    # 라벨 값은 폴더 안에 있는 폴더의 이름 순서대로 지정하되
    # 바이너리 모드인 0 또는 1로 각각 지정한다.
    class_mode='binary'
)

xy_test = test_datagen.flow_from_directory(
    # 첫번 째 인자는 디렉토리 폴더를 받음
    '../data/image/brain/test', # (120,256,256,3)
    # 폴더 안에 있는 전체 이미지 데이터를 사이즈를 바꿈
    target_size=(150,150), # (120,150,150,3)
    batch_size=10, # (전체 데이터/batch_size, batch_size,150,150,3 5차원 배열)
                  # (만약 베치사이즈가 전체 데이터 사이즈보다 크면 4차원 배열 생성됨)
    # 폴더 안에 있는 이미지에 대해서 라벨값을 지정하는데
    # 라벨 값은 폴더 안에 있는 폴더의 이름 순서대로 지정하되
    # 바이너리 모드인 0 또는 1로 각각 지정한다.
    class_mode='binary'
)
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

# 핏제너레이터는 x,y를 나누지 않고 넣음
history = model.fit_generator(
    # steps_per_epoch 는 전체 데이터를 에포치로 나눈 값을 넣는다.
    xy_train, steps_per_epoch=16, epochs=100,
    validation_data=xy_test, validation_steps=4
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 시각화 할 것!!

print('acc :', acc[-1])
print('val_acc :', val_acc[:-1])
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