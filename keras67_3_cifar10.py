#  실습
#  cifer10을 flow 로 구성해서 완성
#  imageDataGenerator / fit_generator
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
print(x_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

datagen = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

"""
def flow(self,
        x,
        y=None,
        batch_size=32,
        shuffle=True,
        sample_weight=None,
        seed=None,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        subset=None):
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPool2D, BatchNormalization, AveragePooling2D

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(10,10),input_shape=(32,32,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides = (2,2),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(5,5),padding='same',activation='relu'))
# model.add(AveragePooling2D((2,2),(2,2)))
model.add(BatchNormalization())
# model.add(Conv2D(filters=4, kernel_size=(3,3),padding='same',activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2),strides = (2,2),padding='same'))
# model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# 핏제너레이터는 x,y를 나누지 않고 넣음

epochs = 5
bath = 500

history = model.fit(datagen.flow(x_train, y_train, batch_size=bath),
                    steps_per_epoch=len(x_train) / bath, epochs=epochs)
# history = model.fit_generator(

#     xy_train, steps_per_epoch=10, epochs=10,validation_data=xy_val,
#     validation_steps=4
# )
"""
here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=bath):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / bath:
           we need to break the loop by hand because
           the generator loops indefinitely
            break
"""
acc = history.history['acc']
loss = history.history['loss']
print('acc :', acc[-1])

#4. Evaluate, predict
evl_loss, evl_acc = model.evaluate(x_test, y_test, batch_size=3)

print("evl_loss : ", evl_loss)
print("evl_acc : ", evl_acc)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2_m1 = r2_score(y_test, y_predict)

print("R2 :", r2_m1 )


# fit.. hist

print(history)
print(history.history.keys()) #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('cnn fashion_mnist')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'acc'])
plt.show()

"""
acc : 0.44001999497413635
evl_loss :  329.9654541015625
evl_acc :  0.1347000002861023
R2 : -0.9208428758122565
<tensorflow.python.keras.callbacks.History object at 0x000001EE7CA82B20>
dict_keys(['loss', 'acc'])
"""