from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# 워드
docs = ["너무 재밋어요", "참 최고에요", "참 잘 만든 영화예요",
        "추천하고 싶은 영화입니다.", "한 번 더 보고 싶네요", '글세요',
        '별로에요', '생각보다 지루해요', "연기가 어색해요",
        '재미없어요', '너무 재미없다', "참 재밋네요", "규현이가 잘생기긴 했어요"]

# 긍정, 부정 라벨 분류
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])


# 워드를 말뭉치 토큰화하는 전처리 단계
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
x = token.texts_to_sequences(docs)
print(x)

# 말뭉치의 유니크한 단어들을 인덱스로 바꾼다.
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='post', maxlen=5)
print(pad_x)
print(type(pad_x))

# 쉐이프 확인
print(pad_x.shape)

# 말뭉치 종류 및 길이
print(np.unique(pad_x))
print(len(np.unique(pad_x)))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D
# 임베딩 모델을 사용한다.
# input_dim는 말뭉치 길이보다 작을 수 없음
# output_dim은 학습하게될 백터 공간의 크기이다.
# input_length 사용하게될 텍스트 단어 수 이다 말뭉치의 단어 인덱스 중 길이가 가장 긴 것보다 길거나 같아야 한다.

model = Sequential()
model.add(Embedding(input_dim=28, output_dim= 11, input_length= 5))
model.add(LSTM(32))
model.add(Dense(1,activation="sigmoid"))
# 파라미터의 수는 input_dim * outpuy_dim 이다.
model.summary()

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["acc"])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1]
print(acc)

# 1.0
