from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters = 10, kernel_size=(2,2), strides=1,    # kernel_size 자르는 사이즈
     padding= "same", input_shape=(10,10,1)))
model.add(MaxPool2D(pool_size=(2,3)))                          # 특성 추출. 
model.add(Conv2D(9, (2,2), padding='valid'))
# model.add(Conv2D(9, (2,3)))
# model.add(Conv2D(8, 2))
model.add(Flatten())
model.add(Dense(1))

model.summary()