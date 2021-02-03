# import os, sys
# from google.colab import drive
# drive.mount('/content/MyDrive')

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten 
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
# from keras.utils import np_utils
import cv2

import gc
from keras import backend as bek
train = pd.read_csv('../data/csv/Dacon2/data/train.csv')

from sklearn.model_selection import train_test_split

x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 28, 28, 1)

# 특성치를 앞에서 전처리에서 빼줌
# 조건에 맞으면 0 틀리면 값을 그대로 둠
x_train = np.where((x_train<=20)&(x_train!=0) ,0.,x_train)

x_train = x_train/255
x_train = x_train.astype('float32')

# 원핫 인코딩
y = train['digit']
y_train = np.zeros((len(y), len(y.unique())))  # 총 행의수 , 10(0~9)
for i, digit in enumerate(y):
    y_train[i, digit] = 1

# 뭐하는거지 왜하는걸까
# 2048,300,300,3의 쉐이프를 0.로 초기화
train_224=np.zeros([2048,300,300,3],dtype=np.float32)

# x_train의 특성치 만큼 반복 이게뭘까.
# 뭔지는 모르겠지만 2048 인덱스만큼 그레이 RGB로 색을 바꿔가면서 그림을 채워넣고있음
for i, s in enumerate(x_train):
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(converted,(300,300),interpolation = cv2.INTER_CUBIC)
    del converted
    train_224[i] = resized
    del resized
    bek.clear_session()
    gc.collect()


from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

# 이미지 범핑 셋팅
# 좌우 위아래 줌 회전 할거 다하고 발리데이션 스플릿
datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # zca_whitening=True,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.15,
        rotation_range = 10,
        validation_split=0.2)
# train_datagen = ImageDataGenerator(
#     rescale = 1./255.,
#                                   #  rotation_range = 10,
#                                    width_shift_range = 0.1,
#                                    height_shift_range = 0.1,
#                                    shear_range = 0.1,
#                                    zoom_range = 0.1
#                                    validation_split=0.1                                   
#                                    )

# train_gen = datagen.flow(x_train,y_train, batch_size=16)
# valid_datagen = ImageDataGenerator(rescale=1./255) 

# 이미지 데이터 범핑
valgen = ImageDataGenerator(
            # featurewise_center=True,
            # zca_whitening=True,
        )
# 모델정의
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
def create_model() :
    # 2020 년 말인가? 네트워크의 넓이 깊이 높이와 엑큐러시 관계에 대해 논문에 실린 이미지네트워크 모델임 
  effnet = tf.keras.applications.EfficientNetB3(
      include_top=True,
      weights=None,
      input_shape=(300,300,3),
      classes=10,
      classifier_activation="softmax",
  )
  model = Sequential()
  model.add(effnet)

    # RMSprop 모름======================================================= 러닝레이트만 알겠음
  model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(lr=initial_learningrate),
              metrics=['accuracy'])
  return model

initial_learningrate=2e-3  
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold

#  RepeatedKFold 같은거 다섯셋 다른거 열셋
kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=40)
cvscores = []
Fold = 1
results = np.zeros((20480,10) )
def lr_decay(epoch):#lrv
    return initial_learningrate * 0.99 ** epoch


test = pd.read_csv('../data/csv/Dacon2/data/test.csv')


x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = np.where((x_test<=20)&(x_test!=0) ,0.,x_test)
# x_test = np.where(x_test>=145,255.,x_test)
x_test = x_test/255
x_test = x_test.astype('float32')

test_224=np.zeros([20480,300,300,3],dtype=np.float32)


for i, s in enumerate(x_test):
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(converted,(300,300),interpolation = cv2.INTER_CUBIC)
    del converted
    test_224[i] = resized
    del resized

bek.clear_session()
gc.collect()



results = np.zeros( (20480,10),dtype=np.float32)


for train, val in kfold.split(train_224): 
    # if Fold<25:
    #   Fold+=1
    #   continue
    
    initial_learningrate=2e-3  
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=50)      
    filepath_val_acc="../data/modelCheckPoint/Dacon2_effic_"+str(Fold)+".ckpt"
    checkpoint_val_acc = ModelCheckpoint(filepath_val_acc, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)

    gc.collect()
    bek.clear_session()
    print ('Fold: ',Fold)
    
    X_train = train_224[train]
    X_val = train_224[val]
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    Y_train = y_train[train]
    Y_val = y_train[val]

    model = create_model()


    training_generator = datagen.flow(X_train, Y_train, batch_size=32,seed=7,shuffle=True)
    validation_generator = valgen.flow(X_val, Y_val, batch_size=32,seed=7,shuffle=True)
    model.fit(training_generator,epochs=150,callbacks=[LearningRateScheduler(lr_decay),es,checkpoint_val_acc],
               shuffle=True, verbose=1,
               validation_data=validation_generator,
               steps_per_epoch =len(X_train)//32
               )
    del X_train
    del X_val
    del Y_train
    del Y_val

    gc.collect()
    bek.clear_session()
    model.load_weights(filepath_val_acc)
    results = results + model.predict(test_224)
    
    Fold = Fold +1


submission = pd.read_csv('../data/csv/Dacon2/data/submission.csv')
submission['digit'] = np.argmax(results, axis=1)
# model.predict(x_test)
submission.head()

submission.to_csv('../data/csv/Dacon2/data/baseline_effic.csv', index=False)
# np.savetxt('/content/MyDrive/My Drive/Colab Notebooks/data/results.csv',results ,delimiter=',')
 
# submission = pd.read_csv('/content/MyDrive/My Drive/Colab Notebooks/data/submission.csv')
# submission['digit'] = np.argmax(results, axis=1)
# submission.head()
# submission.to_csv('/content/MyDrive/My Drive/Colab Notebooks/kfold_effi_5.csv', index=False) 
"""
# ==========================================================================================
"""
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import os
# from tensorflow.keras.optimizers import RMSprop
# # from tensorflow.keras.applications.efficientnet import EfficientNetB7
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten 
# from tensorflow.keras.models import Model
# from tensorflow.keras import optimizers
# import cv2

# import gc
# from keras import backend as bek

# test = pd.read_csv('/content/MyDrive/My Drive/Colab Notebooks/data/test.csv')

# x_test = test.drop(['id', 'letter'], axis=1).values
# x_test = x_test.reshape(-1, 28, 28, 1)
# x_test = np.where((x_test<=20)&(x_test!=0) ,0.,x_test)
# # x_test = np.where(x_test>=145,255.,x_test)
# x_test = x_test/255
# x_test = x_test.astype('float32')

# test_224=np.zeros([20480,300,300,3],dtype=np.float32)


# for i, s in enumerate(x_test):
#     converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
#     resized = cv2.resize(converted,(300,300),interpolation = cv2.INTER_CUBIC)
#     del converted
#     test_224[i] = resized
#     del resized

# bek.clear_session()
# gc.collect()




# #%%

# effnet = tf.keras.applications.EfficientNetB3(
#     include_top=True,
#     weights=None,
#     input_shape=(300,300,3),
#     classes=10,
#     classifier_activation="softmax",
# )



# loaded_model = Sequential()
# loaded_model.add(effnet)


# loaded_model.compile(loss="categorical_crossentropy",
#             optimizer=RMSprop(lr=2e-3),
#             metrics=['accuracy'])

# del x_test
# del test
# results = np.zeros( (20480,10),dtype=np.float16)

# for j in range(50):
#   filepath_val_acc="/content/MyDrive/My Drive/Colab Notebooks/models/effi_model_aug"+str(j+1)+".ckpt"
#   loaded_model.load_weights(filepath_val_acc)
#   results = results + loaded_model.predict(test_224)
  
#   del filepath_val_acc
#   bek.clear_session()
#   gc.collect()
  
# np.savetxt('/content/MyDrive/My Drive/Colab Notebooks/data/results.csv',results ,delimiter=',')  ## 유사도 판정표


# #%% md

# Predict 결과를 앙상블하여 최종적인 예측값 결정

# #%%


# submission = pd.read_csv('/content/MyDrive/My Drive/Colab Notebooks/data/submission.csv')
# submission['digit'] = np.argmax(results, axis=1)
# # model.predict(x_test)
# submission.head()
# submission.to_csv('/content/MyDrive/My Drive/Colab Notebooks/loadtest2.csv', index=False)