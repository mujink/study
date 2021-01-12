
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)             # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)               # (10000, 28, 28) (10000,)

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1).astype('float32')/255.
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1).astype('float32')/255.
# (x_test.reshap(10000, 28, 28, 1))

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           
y_train = y_train.reshape(-1,1)                 
y_test = y_test.reshape(-1,1)                 
y_val = y_val.reshape(-1,1)                 
one.fit(y_train)                          
one.fit(y_test)                          
one.fit(y_val)                          
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
y_val = one.transform(y_val).toarray()


# print(y_train.shape)
# print(y_train.shape)

# print(y_test.shape)
# print(x_train.shape)
# print(x_test.shape)
print("=============")




from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout ,Activation
from tensorflow.keras.models import Sequential, load_model

model = Sequential()
# model.add(Conv2D(filters = 10, kernel_size=(10,10), strides=1,    # kernel_size 자르는 사이즈
#      padding= "same", input_shape=(28,28,1)))
# model.add(MaxPool2D(pool_size=(1,1)))                          # 특성 추출. 
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(10, (2,2), padding='valid'))
# model.add(MaxPool2D(pool_size=(2,2)))                          # 특성 추출. 
# model.add(Activation('relu'))
# model.add(Flatten())                                            # 1dim
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='softmax'))
model = load_model('../data/h5/k51_1_model1.h5')
model.summary()

#3. Compile, train / binary_corssentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = "../data/modelCheckpoint/k45_mnist_{epoch:02d}_{val_loss:.4f}.hdf5"  # 가중치 저장 위치
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
cp = ModelCheckpoint(filepath=(modelpath), monitor='val_loss', save_best_only=True, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_val, y_val), batch_size= 32, callbacks=[early_stopping, cp])
# model.save('../data/h5/k51_1_model2.h5')

# #4. Evaluate, predict
# loss, mae = model.evaluate(x_test, y_test, batch_size=3)

# print("loss : ", loss)
# print("acc : ", mae)

# y_predict = model.predict(x_test)


# from sklearn.metrics import r2_score
# r2_m1 = r2_score(y_test, y_predict)

# print("R2 :", r2_m1 )


# # fit.. hist

# import matplotlib.pyplot as plt
# #한글 폰트 출력 방법
# # import matplotlib.font_manager as fm
# # path = '/Library/Fonts/NanumBarunpenRegular.otf'
# # fontprop = fm.FontProperties(fname=path, size=18)

# plt.figure(figsize=(10,6))          # plot 사이즈
# plt.subplot(2,1,1)              # 2행 1열 중 첫번쨰
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label= 'val_loss')
# plt.grid() # 격자
# plt.rc('font', family='Malgun Gothic')

# # plt.title('비용', fontproperties=fontprop)
# plt.title('비용')
# # plt.title('Cost Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# plt.subplot(2,1,2)              # 2행 2열 중 두번쨰
# plt.plot(hist.history['acc'], marker='.', c='green')
# plt.plot(hist.history['val_acc'], marker='.', c='yellow')
# plt.grid() # 격자

# # plt.title('정확도', fontproperties=fontprop)
# plt.title('정확도')
# # plt.title('Accuracy')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])
# plt.show()


# """
# loss :  0.06191143020987511
# acc :  0.9811000227928162
# """
