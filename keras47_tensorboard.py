"""
tensorboard
cmd 명령어
cd\
study
cd graph
dir/w
tensorboard --logdir=.
웹 키고
http//127.0.0.1:6006
"""


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
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 10, kernel_size=(10,10), strides=1,    # kernel_size 자르는 사이즈
     padding= "same", input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(1,1)))                          # 특성 추출. 
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), padding='valid'))
model.add(MaxPool2D(pool_size=(2,2)))                          # 특성 추출. 
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Flatten())                                            # 1dim
model.add(Dense(20, activation='relu'))
# model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()


#3. Compile, train / binary_corssentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
modelpath = "./modelCheckpoint/k45_mnist_{epoch:02d}_{val_loss:.4f}.hdf5"  # 가중치 저장 위치
es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
cp = ModelCheckpoint(filepath=(modelpath), monitor='val_loss', save_best_only=True, mode='auto')
tb =TensorBoard(log_dir='./graph', histogram_freq=0 , write_images=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_val, y_val), batch_size= 32, callbacks=[es, cp, tb])
""""
log_dir : TensorBoard에서 파싱 할 로그 파일을 저장할 디렉토리의 경로.
histogram_freq : 모델의 계층에 대한 활성화 및 가중치 히스토그램을 계산할 빈도 (에포크 단위).
	0으로 설정하면 히스토그램이 계산되지 않습니다.
	히스토그램 시각화를 위해 정렬 데이터 (또는 분할)를 지정해야합니다.
write_graph : 텐서 보드에서 그래프를 시각화할지 여부.
	write_graph가 True로 설정되면 로그 파일이 상당히 커질 수 있습니다.
write_images : TensorBoard에서 이미지로 시각화하기 위해 모델 가중치를 작성할지 여부.
update_freq : 'batch'또는 'epoch'또는 정수. (log 업데이트 주기)
profile_batch : 컴퓨팅 특성을 샘플링하기 위해 배치를 프로파일 링합니다.
     기본으로 두 번째 배치를 프로파일 링합니다.
	profile_batch = 0 => 비활성화
embeddings_freq : 임베딩 레이어가 시각화되는 주기(에포크 단위)
	0으로 설정하면 임베딩이 시각화되지 않습니다.
embeddings_metadata :이 임베딩 레이어에 대한 메타 데이터가 저장되는 파일 이름에 레이어 이름을 매핑하는 사전입니다. 


"""

#4. Evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=3)

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
loss :  0.06191143020987511
acc :  0.9811000227928162
"""
