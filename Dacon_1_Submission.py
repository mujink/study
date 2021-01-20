# 훈련된 모델로 테스트 파일의 출력을 보려함.
# 테스트의 1일치 파일로 다음 2일치의 출력을 출력함
# 80 개의 테스트 파일을 9번 프레딕트함
# 일딴 출력만 나와줘라.
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model

x = np.load('../data/csv/Dacon/np/TrainDb_X.npy',allow_pickle=True)
y = np.load('../data/csv/Dacon/np/TrainDb_Y.npy',allow_pickle=True)

# train_test_split ==========================================================================================
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = False)#, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = False)#, random_state=1)

from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# === Submission 준비 ============================================

#  sample_submission csv 불러오기
submission = pd.read_csv('../data/csv/Dacon/sample_submission.csv')

#  모델 불러오기
model = load_model('../data/h5/Dacon_soler_cell.hdf5')

#  테스트셋 불러오기
fitset = pd.read_csv('../data/csv/Dacon/preprocess_csv/TestDbSet2.csv', encoding='ms949', index_col=0)

# === Submission 진행 ============================================

quantiles1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
quantiles = 9


from tensorflow.keras.backend import mean, maximum

def quantile_loss(q, y, pred):
  err = (y-pred)
  return mean(maximum(q*err, (q-1)*err), axis=-1)

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

modelpath = "../data/h5/Dacon_soler_cell.hdf5"
es = EarlyStopping(monitor = 'val_loss',patience=20, mode="min")
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=15, factor=0.5, verbose=1)

import matplotlib.pyplot as plt

# for j in range(quantiles) :
for j in quantiles1 :
    model.compile(loss=lambda y,pred: quantile_loss(j,y,pred), optimizer='adam')
    hist = model.fit (x_train, y_train,  epochs= 10000, batch_size=128, verbose=1, validation_data=(x_val , y_val),callbacks=[es,reduce_lr], shuffle=False)
    temp = model.predict(fitset)

    col = 'q_' + str(j)
    submission.loc[submission.id.str.contains("Day7"),col] = temp[:,0]
    submission.loc[submission.id.str.contains("Day8"),col] = temp[:,1]


submission.to_csv('../data/csv/submission_v4.csv', index=False)

#  엥 로스 어디까지 떨어지는거야


"""
from lightgbm import LGBMRegressor

# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
    
    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)                   
                         
                         
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)

    # (b) Predictions
    pred = pd.Series(model.predict(X_test).round(2))
    return pred, model

# Target 예측

def train_data(X_train, Y_train, X_valid, Y_valid, X_test):

    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        print(q)
        pred , model = LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test)
        LGBM_models.append(model)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)

    LGBM_actual_pred.columns=quantiles
    
    return LGBM_models, LGBM_actual_pred

# Target1
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)
results_1.sort_index()[:48]

# Target2
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_test)
results_2.sort_index()[:48]

# 값 넣기 
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
submission

#  파일 세이브 
submission.to_csv('./data/submission_v3.csv', index=False)
"""