import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from xgboost import XGBClassifier

# 불러오기
train = pd.read_csv('../data/csv/Dacon2/data/train.csv',header = 0)
test = pd.read_csv('../data/csv/Dacon2/data/test.csv',header = 0)
# ==============================그림보기==========================
# 인덱스
idx = 318
# 트레인 데이터의 인덱스 길이만큼의 데이터에 대해 0번 컬럼의 값을 28,28로 쉐이프하여 이미지로 초기화
img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
# 트레인 데이터의 인덱스 길이만큼 타겟을 디지트로 초기화
digit = train.loc[idx, 'digit']
# 트레인 데이터의 인덱스 길이만큼 레터를 레터로 초기화
letter = train.loc[idx, 'letter']

# 타이틀을 인덱스, 디지트, 레터로 함
plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
# 인덱스 번째의 쉐이프한 이미지를 이미지로 출력
plt.imshow(img)
# 보기
plt.show()
# ==============================데이터 전처리==========================

# 트래인의 아이디, 디지트, 레터 컬럼을 제거한 나머지 컬럼의 값을 축변경하여 초기화
x_train = train.drop(['id', 'digit', 'letter'], axis=1).values

from sklearn.decomposition import PCA

pca = PCA()
x2 = pca.fit_transform(x_train)
print(x2.shape)                 # (70000, 784)

# 컬럼을 압축한 컬럼의 변화 비율을 확인
# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))             # 1.000000000000002

# pca = PCA(130)
# pca.fit(x_train)
# cumsum = 누적합계
cumsum = np.cumsum(pca.explained_variance_ratio_)
print("누계 :", cumsum)
d = np.argmax(cumsum >= 0.99)+1
print("cumsum >= 0.95", cumsum >= 0.95)
print("d :", d)

# 시엔엔 쉐이프로 변경
x_train = x_train.reshape(-1, 28,28,1)
# 0~1 사이로 값바꿈
x_train = x_train/255


# 와이값은 트레인의 디지트 컬럼만을 참조함
y = train['digit']
# 와이 트레인은 와이의 길이를 행으로, 와이의 중복값을 제거한 나머지 값의 길이를 열로하는 배열을 만들고
# 0으로 초기화
# 아마 원핫인코딩을 이렇게 하려하는 듯 왜냐하면 와이값의 유니크를 컬럼으로 잡고 와이 길이만큼 0으로 준걸 보면
# 나중에 유니크 길이만큼 1을 넣으려하는 듯
y_train = np.zeros((len(y), len(y.unique())))

# 와이트래인을 원 핫 인코딩함
for i, digit in enumerate(y):
    y_train[i, digit] = 1




from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.5, shuffle = True, random_state=1)


#  모델을 정의하고 변수 트래인을 인자로 받음
def create_cnn_model(x_train):
    
    # 인풋 쉐이프 레이어
    inputs = tf.keras.layers.Input(x_train.shape[1:])
    # 아웃풋을 전처리하여 다음레이어에 전달해주는 엑티베이션 레이어로 시엔엔 모델의 각층에 삽입해 explode또는 vanish 을 잡음.
    bn = tf.keras.layers.BatchNormalization()(inputs)
    conv = tf.keras.layers.Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu')(bn)
    bn = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

    bn = tf.keras.layers.BatchNormalization()(pool)
    conv = tf.keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    bn = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

    flatten = tf.keras.layers.Flatten()(pool)

    bn = tf.keras.layers.BatchNormalization()(flatten)
    dense = tf.keras.layers.Dense(1000, activation='relu')(bn)

    bn = tf.keras.layers.BatchNormalization()(dense)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(bn)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)


    return model
# ==================================================================================================


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = "../data/modelCheckPoint/Dacon2_{epoch:02d}_{val_accuracy:.2f}.hdf5"  # 가중치 저장 위치
early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, mode='max')
cp = ModelCheckpoint(filepath=(modelpath), monitor='val_accuracy', save_best_only=True, mode='auto')
# 이거 추가됨
# 모니터 벨류가 patience 회 개선이 없으면 러닝레이factor를 % 만큼 감소시켜준다.
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',patience=10, factor=0.5, verbose=1)
# # ==================================================================================================

model = create_cnn_model(x_train)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=300,validation_data=(x_val,y_val), batch_size=256 ,callbacks=[early_stopping,cp,reduce_lr])


x_test = test.drop(['id', 'letter'], axis=1).values
print(x_test.shape)
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test/255

submission = pd.read_csv('../data/csv/Dacon2/data/submission.csv')
submission['digit'] = np.argmax(model.predict(x_test), axis=1)
print(submission.head())

# submission.to_csv('../data/csv/Dacon2/data/baseline.csv', index=False)
