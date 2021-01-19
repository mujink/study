import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model

df_test = np.array([])

#  sample_submission csv 불러오기
submission = pd.read_csv('../data/csv/Dacon/sample_submission.csv')

#  모델 불러오기
model = load_model('../data/h5/Dacon_soler_cell.hdf5')

#  테스트셋 불러오기
fitset = pd.read_csv('../data/csv/Dacon/preprocess_csv/TestDbSet2.csv', encoding='ms949', index_col=0)

quantiles = [1,2,3,4,5]

for q in quantiles:
        print(q)
        # df_test.iloc[:,q*2-2:q*2-1] = model.predict(fitset)
        df_test[:,(q*2)-2:(q*2)-1] = model.predict(fitset)
        # submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = y_predict.sort_index().values
        # submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = y_predict.sort_index().values

        # df_test.transpose.append(y_predict)


print(df_test)
#  81개 값 저장
# submission.to_csv('../data/csv/submission_v5.csv', index=False)

# y_pred = pd.DataFrame(df_test)
df_test.to_csv('../data/csv/submission_v4.csv', index=False)



# #  함수 정의
# def split_xy(seq,x_size,x_col_start, x_col_end ,y_size,y_col_start,y_col_end):
#     print(range(len(seq)-x_size-1))                             
#     print(seq.shape)                                            
#     x=[]
#     y=[]
#     for i in range(len(seq)-x_size-y_size+1):                          
#         xi = seq[i:(i+x_size),x_col_start-1:x_col_end].astype('float32')   
#         yi = seq[(i+x_size):(i+x_size+y_size),y_col_start-1:y_col_end].astype('float32')       
#         x.append(np.array(xi))          
#         y.append(np.array(yi))          
#     print(np.array(x).shape)
#     print(np.array(y).shape)
#     return np.array(x),  np.array(y)


# # 값 넣기 
# submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
# submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
# submission

# #  파일 세이브 
# submission.to_csv('./data/submission_v3.csv', index=False)



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