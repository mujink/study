# 실습
# cifar10으로 vgg16 넣어서 만들 것!!
from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, AveragePooling2D, BatchNormalization, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)             # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)               # (10000, 32, 32, 3) (10000, 1)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           
y_train = y_train.reshape(-1,1)                 
y_test = y_test.reshape(-1,1)                 
one.fit(y_train)                          
one.fit(y_test)                          
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()


from sklearn.model_selection import train_test_split

x_train = x_train/255.

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)


vgg16 = VGG16(weights="imagenet", include_top= False, input_shape=(32,32,3))
# vgg16 = VGG19()

# include_top이 False여야 인풋쉐이프를 바꿀 수 있음
# vgg16.trainable = True  # 해당 레이어 가중치 동결하지 않음
vgg16.trainable = True # 해당 레이어 가중치 동결

# =================================================================
# Total params: 20,346,144
# Trainable params: 20,346,044
# Non-trainable params: 100
# _________________________________________________________________
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

model = Sequential()
model.add(vgg16)
model.add(Reshape(target_shape=(16,16,2)))
model.add(Conv2D(filters = 50, kernel_size=(2,2), activation="swish",   # kernel_size 자르는 사이즈
     padding= "same"))
model.add(AveragePooling2D((2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(4096, activation="swish"))
model.add(Dense(4096, activation="swish"))
model.add(Dense(10, activation="softmax"))
vgg16.summary()

print("그냥 가중치의 수:",len(vgg16.weights))           #32
print("동결후 훈련되는 가중치의 수",len(vgg16.trainable_weights)) #6

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# modelpath = "./data/modelCheckpoint/k76_VGG16-cifar10_{epoch:02d}_{val_loss:.4f}.hdf5"  # 가중치 저장 위치
early_stopping = EarlyStopping(monitor='val_acc', patience=100, mode='min')
lr = ReduceLROnPlateau(monitor='val_acc',patience=20, factor=0.7, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=1000, verbose=1, validation_data=(x_val, y_val), batch_size= 256,
                callbacks=[early_stopping, lr])

loss1, acc1 = model.evaluate(x_test, y_test)
print("loss :", loss1)
print("acc :", acc1)

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('cnn fashion_cifer10')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss','val_loss', 'acc','val_acc'])
plt.show()

# loss : 1510.2994384765625
# acc : 0.14640000462532043