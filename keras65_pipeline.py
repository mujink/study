# 61번을 파이프라인으로 구성

#  가중치 저장
# 모델 세이브() 사용
# 피클 사용

# 61 copy해서
# model.cv_resualts 를 출력

# 콜백, 발리데이션, 에포
import pandas as pd

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)
# print(x_test.shape)
# 1. 데이터 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255 
x_test = x_test.reshape(10000, 28*28).astype('float32')/255 

# 2. 모델
def build_model(drop=0.5, optimizer='adam'):
    input = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(input)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    output = Dense(10, activation='softmax', name='outputs')(x)

    model = Model(inputs = input, outputs = output)
    model.compile(optimizer=optimizer, metrics=['acc'],
                loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1,0.2,0.3]
    validation_split = [0.1,0.2,0.3]
    epochs = [10,20,30]
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout,'epochs':epochs,'validation_split':validation_split}

hyperparameters = create_hyperparameters()

# 랜덤 서치 사용하기위해 케라스 랩핑



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

modelpath = "../data/modelCheckpoint/keras61_{epoch:02d}_{val_loss:.4f}.hdf5"  # 가중치 저장 위치
cp = ModelCheckpoint(filepath=(modelpath), monitor='val_loss', save_best_only=True, mode='auto')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=10, factor=0.3, verbose=1)


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2= KerasClassifier(build_fn=build_model, verbose=1, epochs = 100, validation_split=0.2)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2,hyperparameters,cv=3)
# search = GridSearchCV(model2, hyperparameters, cv=3)

from sklearn.pipeline import Pipeline, make_pipeline
pipe = Pipeline([("scaler", MinMaxScaler()),("Rand", search)])


search.fit(x_train,y_train, verbose=1,callbacks=[early_stopping,cp,reduce_lr])
result = pd.DataFrame(search.cv_results_)
print(search.best_params_)
print("resualt :",result)
print(search.best_estimator_)
print(search.best_score_)
acc = search.score(x_test,y_test)
print('최종 스코어 :', acc)

import pickle
pickle.dump(model2, open('../data/xgb_save/keras64_save_pickel.dat','wb'))
print("======================pickle load=========================")
# model2 = pickle.load(open('../data/xgb_save/keras64_save_pickel.dat','rb'))

"""
[108 rows x 18 columns]
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000002858F0DE130>
0.9758166551589966
200/200 [==============================] - 0s 2ms/step - loss: 0.0657 - acc: 0.9806
최종 스코어 : 0.9805999994277954
======================pickle load=========================
"""