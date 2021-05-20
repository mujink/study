# 데이터 셋을 불러와 정렬하는 하는 작업
"""
# train set의 과정
# target1 = n+1일 발전량
# target2 = n+2일 발전량
# null 값이 있는 1일치 제거 (-48)
# train shape (52464, 9)


# test set의 과정
# test shape (3888, 7)
# 각 81일치 마지막 날 (7일)로 묶음.

# 모델은 7컬럼 입력 후 D+1, D+2를 출력할 예정임.
"""

import pandas as pd
import numpy as np


File = 81
df_test = []

# 함수 정의
def Add_features(data):
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    data.insert(1,"Hour_Minute",data["Hour"] * 2 + data["Minute"] // 30)
    return data

def preprocess_data(data, is_train=True):
    data = Add_features(data)
    temp = data.copy()
    temp = temp[['Hour_Minute', 'GHI', 'Hour','Td','T-Td', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train==False:
        
        temp = temp[['Hour_Minute', 'GHI', 'Hour','Td','T-Td', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

        return temp.iloc[-48:, :]



#  train csv 불러오기
file_train = pd.read_csv('../data/csv/Dacon/train/train.csv')
x_train = preprocess_data(file_train, is_train=True)


# test csv 81개 파일 불러오기
for i in range(File) :
      file_path = '../data/csv/Dacon/test/' + str(i) + '.csv'
      temp = pd.read_csv(file_path,encoding='ms949')
      temp = preprocess_data(temp, is_train= False)
      df_test.append(temp)

X_test = pd.concat(df_test)

print(X_test.shape)
print(x_train.shape)

# 저장 ===================================================================================
X_test.to_csv('../data/csv/Dacon/preprocess_csv/TestDbSet.csv', encoding='ms949', sep=",")
x_train.to_csv('../data/csv/Dacon/preprocess_csv/TrainDbSet.csv', encoding='ms949', sep=",")
