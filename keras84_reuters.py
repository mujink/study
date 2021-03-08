from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train[0], type(x_train[0]))
print(y_train[0])
print("=================")
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

print("뉴스기사 최대길이 :", max(len(i) for i in x_train) )
print("뉴스기사 평균길이 :", sum(map(len, x_train))/ len(x_train))


# y 분포
unique_elemnets, counts_elements = np.unique(y_train, return_counts=True)
print("y 분포", dict(zip(unique_elemnets, counts_elements)))
print("===========================================")
# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()


# x 의 단어들 분포
word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index))
# plt.hist(y_train, bins=46)
# plt.show()

# 키와 벨류를 교체!!
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

print(index_to_word)
print(index_to_word[1])
print(index_to_word[30979])
print(len(index_to_word))

print(x_train[0])
print(' '.join([index_to_word[index]for index in x_train[0]]))

category1 = np.max(y_train) + 1
print("y categorys : ", category1)

y_bunpo  = np.unique(y_train)
print(y_bunpo)

# from sklearn.preprocessing import OneHotEncoder
# one = OneHotEncoder()                           
# y_train = y_train.reshape(-1,1)                 
# y_test = y_test.reshape(-1,1)                 
# one.fit(y_train)                          
# one.fit(y_test)                          
# y_train = one.transform(y_train).toarray()
# y_test = one.transform(y_test).toarray()

x_train = pad_sequences(x_train, maxlen= 500)
x_test = pad_sequences(x_test, maxlen= 500)
print(y_train.shape, y_test.shape)
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

print(np.unique(x_train))
print(len(np.unique(x_train)))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D
# 임베딩 모델을 사용한다.
# input_dim는 말뭉치 길이보다 작을 수 없음
# output_dim은 학습하게될 백터 공간의 크기이다.
# input_length 사용하게될 텍스트 단어 수 이다 말뭉치의 단어 인덱스 중 길이가 가장 긴 것보다 길거나 같아야 한다.

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=147))
model.add(LSTM(16))
# model.add(Flatten())
model.add(Dense(100,activation="swish"))
model.add(Dense(46,activation="softmax"))
# 파라미터의 수는 input_dim * outpuy_dim 이다.
model.summary()

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["acc"])
model.fit(x_train, y_train, epochs=15)

acc = model.evaluate(x_test, y_test)[1]
print(acc)

# 0.6780943870544434

# # 워드를 말뭉치 토큰화하는 전처리 단계
# token = Tokenizer()
# token.fit_on_texts(docs)
# print(token.word_index)
# x = token.texts_to_sequences(docs)
# print(x)

# # 말뭉치의 유니크한 단어들을 인덱스로 바꾼다.
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# pad_x = pad_sequences(x, padding='post', maxlen=5)
# print(pad_x)

# # 쉐이프 확인
# print(pad_x.shape)

# # 말뭉치 종류 및 길이
# print(np.unique(pad_x))
# print(len(np.unique(pad_x)))

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D
# # 임베딩 모델을 사용한다.
# # input_dim는 말뭉치 길이보다 작을 수 없음
# # output_dim은 학습하게될 백터 공간의 크기이다.
# # input_length 사용하게될 텍스트 단어 수 이다 말뭉치의 단어 인덱스 중 길이가 가장 긴 것보다 길거나 같아야 한다.

# model = Sequential()
# model.add(Embedding(input_dim=28, output_dim= 11, input_length= 5))
# model.add(LSTM(32))
# model.add(Dense(1,activation="sigmoid"))
# # 파라미터의 수는 input_dim * outpuy_dim 이다.
# model.summary()

# model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["acc"])
# model.fit(pad_x, labels, epochs=100)

# acc = model.evaluate(pad_x, labels)[1]
# print(acc)

# # 1.0
