from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)

print(x_train[0], type(x_train[0]))
# print(y_train[0])
print("=================")
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# (25000,) (25000,)
# (25000,) (25000,)


x_train = pad_sequences(x_train, maxlen= 1500)
x_test = pad_sequences(x_test, maxlen= 1500)
print(y_train.shape, y_test.shape)
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

print(np.unique(x_train))
print(len(np.unique(x_train)))

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D, BatchNormalization, LeakyReLU, MaxPool1D, PReLU
# 임베딩 모델을 사용한다.
# input_dim는 말뭉치 길이보다 작을 수 없음
# output_dim은 학습하게될 백터 공간의 크기이다.
# input_length 사용하게될 텍스트 단어 수 이다 말뭉치의 단어 인덱스 중 길이가 가장 긴 것보다 길거나 같아야 한다.

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=1500))
model.add(Conv1D(64,10))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(PReLU())
model.add(Conv1D(32,5))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(PReLU())
model.add(Flatten())
model.add(Dense(100,activation="swish"))
model.add(Dense(46,activation="softmax"))
# 파라미터의 수는 input_dim * outpuy_dim 이다.
model.summary()

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["acc"])
model.fit(x_train, y_train, epochs=50, batch_size=32)

acc = model.evaluate(x_test, y_test)[1]
print(acc)


# 리키렐루 학원컴
# 0.8488399982452393
# 리키렐루 집컴
# 0.8149999976158142
# 피렐루
# 0.8624399900436401
# 0.8689200282096863
# 메시
# 0.8596400022506714
