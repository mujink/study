#  가중치 저장
# 모델 세이브() 사용
# 피클 사용

# 61 copy해서
# model.cv_resualts 를 출력

# 콜백, 발리데이션, 에포

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

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
    batches = [50,300]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1,0.2]
    validation_split = [0.1,0.2,0.3]
    epochs = [3,4,5]
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout,'epochs':epochs,'validation_split':validation_split}

hyperparameters = create_hyperparameters()
model2 = build_model()

# 랜덤 서치 사용하기위해 케라스 랩핑


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

modelpath = "../data/modelCheckpoint/keras64_save_{epoch:02d}_{val_loss:.4f}.hdf5"  # 가중치 저장 위치
cp = ModelCheckpoint(filepath=(modelpath), monitor='val_loss', save_best_only=True, mode='auto')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=10, factor=0.3, verbose=1)


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

model2= KerasClassifier(build_fn=build_model, verbose=1, epochs = 100, validation_split=0.2)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = GridSearchCV(model2,hyperparameters)
# search = GridSearchCV(model2, hyperparameters, cv=3)



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
{'batch_size': 300, 'drop': 0.1, 'epochs': 5, 'optimizer': 'adam', 'validation_split': 0.1}
resualt :      mean_fit_time  std_fit_time  mean_score_time  std_score_time  ... split4_test_score mean_test_score std_test_score rank_test_score
0         9.873026      0.103976         0.493646        0.058033  ...          0.968000        0.971583       0.002538              34
1         9.097939      0.071595         0.522330        0.060941  ...          0.965417        0.967317       0.003861              67
2         8.418196      0.077786         0.500094        0.049358  ...          0.970500        0.971083       0.002698              36
3         6.926783      0.147771         0.487013        0.008191  ...          0.970417        0.971650       0.003024              31
4         6.500530      0.166241         0.479386        0.008325  ...          0.970417        0.970483       0.002296              40
..             ...           ...              ...             ...  ...               ...             ...            ...             ...
103       4.485731      0.030736         0.212976        0.000711  ...          0.975833        0.974050       0.001974               5
104       4.385599      0.357184         0.215952        0.003036  ...          0.972000        0.972083       0.001331              24
105       4.874380      0.024931         0.214389        0.002789  ...          0.318083        0.327983       0.018085              91
106       4.608056      0.042608         0.217031        0.002793  ...          0.277583        0.299733       0.037168              93
107       4.277087      0.033232         0.381765        0.334008  ...          0.187000        0.232533       0.033769              97

[108 rows x 18 columns]
<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000026FC9848070>
0.9750499963760376
34/34 [==============================] - 0s 3ms/step - loss: 0.0663 - acc: 0.9807
최종 스코어 : 0.9807000160217285
======================pickle load=========================
"""