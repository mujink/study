import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)
# print(x_test.shape)
# 1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255 
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255 

# 2. 모델
def build_model(drop=0.5, optimizer='adam', kernel_size=(2,2),activation='activation'):
    input = Input(shape=(28,28,1), name='input')
    x = Conv2D(64,(2,2),activation='relu',padding='same', name='hidden1')(input)
    x = BatchNormalization()(x)
    x = MaxPool2D(5,padding='same')(x)
    x = Dropout(drop)(x)
    x = Conv2D(32,2, activation='relu',padding='same', name='hidden2')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Conv2D(16,2, activation='relu',padding='same', name='hidden4')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Conv2D(8,2, activation='relu',padding='same',name='hidden5')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(3,padding='same')(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
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
    activation = ['relu','linear']
    kernel_size= [(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10)]
    dropout = [0.1,0.2,0.3]
    return {"batch_size" : batches, "optimizer" : optimizers, "activation" : activation,
            "kernel_size" : kernel_size, "drop" : dropout}

hyperparameters = create_hyperparameters()
model2 = build_model()

# 랜덤 서치 사용하기위해 케라스 랩핑
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2= KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2,hyperparameters,cv=3 )
# search = GridSearchCV(model2, hyperparameters, cv=3)

search.fit(x_train,y_train, verbose=1 )
print(search.best_params_)
# {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 50}
# {'optimizer': 'adam', 'kernel_size': (8, 8), 'drop': 0.1, 'batch_size': 20, 'activation': 'linear'}
print(search.best_estimator_)
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x00000204275CD220>
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001ACA9F2D4F0>
# 출력이 안나옴
print(search.best_score_)
# 0.9587166706720988
acc = search.score(x_test,y_test)
print('최종 스코어 :', acc)
# 최종 스코어 : 0.9624999761581421
# 최종 스코어 : 0.9646999835968018
