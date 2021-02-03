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
idx = 318
img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
digit = train.loc[idx, 'digit']
letter = train.loc[idx, 'letter']

plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
plt.imshow(img)
plt.show()
# ==============================데이터 전처리==========================
x_test = test.drop(['id', 'letter'], axis=1).values
x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = x_train/255
x_test = x_test/255

# from sklearn.decomposition import PCA

# pca = PCA(400)
# x_train = pca.fit_transform(x_train)
# x_test = pca.fit_transform(x_test)
# print(x_train.shape)                 # (70000, 784)


# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("누계 :", cumsum)
# d = np.argmax(cumsum >= 0.99)+1
# print("cumsum >= 0.95", cumsum >= 0.95)
# print("d :", d)
x_train = x_train.reshape(-1, 28, 28,1)
x_test = x_test.reshape(-1, 28, 28, 1)



# 와이트래인을 원 핫 인코딩함

y = train['digit']

y_train = np.zeros((len(y), len(y.unique())))

for i, digit in enumerate(y):
    y_train[i, digit] = 1

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)


from keras.preprocessing.image import ImageDataGenerator
idg = ImageDataGenerator(rotation_range=200)
idg2 = ImageDataGenerator()

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
train_generator = idg.flow(x_train,y_train)
valid_generator = idg2.flow(x_val,y_val)
test_generator = idg2.flow(x_test,shuffle=False)
print(len(train_generator))
print(len(valid_generator))
print(len(test_generator))

#  모델을 정의하고 변수 트래인을 인자로 받음
def create_cnn_model(x_train):
    
    inputs = tf.keras.layers.Input(x_train.shape[1:])
    bn = tf.keras.layers.BatchNormalization()(inputs)
    conv = tf.keras.layers.Conv2D(128, kernel_size=10, strides=1, padding='same', activation='relu')(bn)
    pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)
    bn = tf.keras.layers.BatchNormalization()(pool)
    conv = tf.keras.layers.Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu')(bn)
    pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

    bn = tf.keras.layers.BatchNormalization()(pool)
    conv = tf.keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    bn = tf.keras.layers.BatchNormalization()(conv)
    pool = tf.keras.layers.MaxPooling2D((2, 2))(bn)

    conv = tf.keras.layers.Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(pool)
    pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

    flatten = tf.keras.layers.Flatten()(pool)

    bn = tf.keras.layers.BatchNormalization()(flatten)
    dense = tf.keras.layers.Dense(1000, activation='relu')(bn)
    dense = tf.keras.layers.Dense(600, activation='relu')(dense)
    dense = tf.keras.layers.Dense(300, activation='relu')(dense)

    bn = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.Dense(100, activation='relu')(bn)

    outputs = tf.keras.layers.Dense(10, activation='softmax')(dense)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)


    return model
# ==================================================================================================


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = "../data/modelCheckPoint/Dacon2_{epoch:02d}_{val_accuracy:.2f}.hdf5"  # 가중치 저장 위치
early_stopping = EarlyStopping(monitor='val_accuracy', patience=300, mode='max')
cp = ModelCheckpoint(filepath=(modelpath), monitor='val_accuracy', save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',patience=20, factor=0.5, verbose=1)
# # ==================================================================================================
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
optimizer = Adam(lr=0.005)

model = create_cnn_model(x_train)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(train_generator, epochs=1000,validation_data=valid_generator, batch_size=256 ,callbacks=[early_stopping,cp,reduce_lr])




submission = pd.read_csv('../data/csv/Dacon2/data/submission.csv')
submission['digit'] = np.argmax(model.predict(x_test), axis=1)
print(submission.head())

submission.to_csv('../data/csv/Dacon2/data/baseline.csv', index=False)
