# 나를 찍어서 내가 남자인지 여자인지 확인
# 액큐러시 

#  실습
#  imageDataGenerator fit_gerator 사용
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPool2D, Activation, BatchNormalization, AveragePooling2D

import cv2
import matplotlib.pyplot as plt

# Read image
# img_path = '../data/image/me.PNG'
img_path = '../data/image/me2.PNG'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_l = img.copy()

input_img = cv2.resize(img_l, (100, 100))
input_img = (input_img / 255.).astype(np.float32)
input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)

# Display image
plt.imshow(input_img)
plt.show()
x_pred = np.array(input_img[:,:,0].reshape(1,100,100,1))
y_pred = 1.0
print(x_pred.shape)

# Fetch image pixel data to numpy array
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
    validation_split= 0.1,
    # 사진을 휘거나 늘어뜨림
    # shear_range=0.5,
    # 이동 후 빈자리를 nearest 모드를 한다.
    fill_mode='nearest'
)

# flow 또는 flow_from_directory(폴더명을 카테고리로 지정한 경우 폴더명이 Y라벨이됨)
# 이미지 플로트 하는 걸 과제로 낼거임

# test_datagen에 대해서 프롤우 프럼 디렉토리를 다음과 같이한다.
xy_train = train_datagen.flow_from_directory(

    '../data/image/data2/', 
    # '../data/image/data2/m',

    # '../data/image/data2_generator/', 

    color_mode ='grayscale',
    target_size=(100,100),
    batch_size=17103,
    class_mode='binary',
    subset="training",
    # save_to_dir='../data/image/data2_generator/fm'
    # save_to_dir='../data/image/data2_generator/m'

)

xy_val = train_datagen.flow_from_directory(

    '../data/image/data2/',
    # '../data/image/data2_generator/', 

    color_mode ='grayscale',
    target_size=(100,100),
    batch_size=17103,

    class_mode='binary',
    subset="validation",
    # save_to_dir='../data/image/data2_generator/m'
)


np.save('../data/image/data2_npy/keras67_4_train_x.npy', arr=xy_train[0][0])
np.save('../data/image/data2_npy/keras67_4_train_y.npy', arr=xy_train[0][1])
np.save('../data/image/data2_npy/keras67_4_test_x.npy', arr=xy_val[0][0])
np.save('../data/image/data2_npy/keras67_4_test_y.npy', arr=xy_val[0][1])

x_train= np.load('../data/image/data2_npy/keras67_4_train_x.npy')
y_train= np.load('../data/image/data2_npy/keras67_4_train_y.npy')
print(x_train.shape)

plt.title(y_train[0])
plt.imshow(x_train[0])
plt.show()

plt.title(y_train[5])
plt.imshow(x_train[5])
plt.show()

x_val= np.load('../data/image/data2_npy/keras67_4_test_x.npy')
y_val= np.load('../data/image/data2_npy/keras67_4_test_y.npy')
print(x_val.shape)
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit

# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3,shuffle = True, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1,shuffle = True, random_state=1)
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
model = Sequential()
# model.add(MaxPool2D(pool_size=(2, 2),input_shape=(128,128,1)))
model.add(Conv2D(filters=16, kernel_size=(5,5),input_shape=(100,100,1),padding='same',activation='relu'))
model.add(AveragePooling2D((2,2),(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=8, kernel_size=(2,2),padding='same',activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=4, kernel_size=(2,2),padding='same',activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(filters=2, kernel_size=(2,2),padding='same',activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(Conv2D(filters=50, kernel_size=(2,2),padding='same',activation='relu'))
# model.add(AveragePooling2D(pool_size=(2,2),padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(Conv2D(filters=4, kernel_size=(2,2),padding='same',activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2),padding='same'))
# model.add(BatchNormalization())
model.add(Flatten())
# model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

modelpath = "../data/modelCheckpoint/keras_67_my_{epoch:02d}_{val_loss:.4f}.hdf5"  # 가중치 저장 위치
early_stopping = EarlyStopping(monitor='val_acc', patience=30, mode='min')
cp = ModelCheckpoint(filepath=(modelpath), monitor='val_acc', save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_acc',patience=10, factor=0.5, verbose=1)

history = model.fit( 
    x_train,y_train, callbacks=[early_stopping,reduce_lr,cp], epochs=2000, batch_size=10, validation_data=(x_val,y_val)
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

y_predict = model.predict(x_pred)
print(y_predict)
# 시각화 할 것!!



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
plt.plot(history.history['acc'], marker='.', c='green', label='acc')
plt.plot(history.history['val_acc'], marker='.', c='yellow', label= 'val_mval_accae')
plt.grid() # 격자

plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

