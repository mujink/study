  
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,LSTM,Conv1D,MaxPool1D,Dropout,Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# # 함수정의
# def split_x(seq,size,col):
#     aaa=[]
#     for i in range(len(seq)-size+1):
#         subset = seq[i:(i+size),0:col].astype('float32')
#         aaa.append(np.array(subset))
#     return np.array(aaa)


# # 콤마 제거, 데이터 타입 변경
# def str_to_float(input_str):
#     temp = input_str
#     if temp[0]!='-':
#         temp = input_str.split(',')
#         sum = 0
#         for i in range(len(temp)):
#             sum+=float(temp[-i-1])*(10**(i*3))
#         return sum
#     else:
#         temp=temp[1:]
#         temp = input_str.split(',')
#         sum = 0
#         for i in range(len(temp)):
#             sum+=float(temp[-i-1])*(10**(i*3))
#         return -sum         


# datasets = pd.read_csv("../data/csv/samsung.csv",encoding='cp949',index_col=0)
# datasets1 = pd.read_csv("../data/csv/삼성전자2.csv",encoding='cp949',index_col=0)

# #전처리 
# #1-1 분할(결측치가 있는 3개의행 제거)
# datasets_1 = datasets.iloc[:662,:]
# datasets_2 = datasets.iloc[665:,:]

# # 행 제거
# datasets_1 =  datasets_1.drop(['2021-01-13'])
# datasets1 =  datasets1.iloc[:2,:]

# # 열 제거
# del datasets1["전일비"]
# del datasets1["Unnamed: 6"]

# # 합치기
# datasets = pd.concat([datasets1,datasets_1,datasets_2,])

# #필요한 열에 타입변환 및 순서 바꿈
# # str -> florat        

# for j in [0,1,2,3,5,6,8,9,10,11,12]:
#     for i in range(len(datasets.iloc[:,j])):
#         datasets.iloc[i,j] = str_to_float(datasets.iloc[i,j])
#     print(datasets.iloc[:0,j], j)
# """
# print(datasets.isnull().sum())

# 시가        0
# 고가        0
# 저가        0
# 종가        0
# 등락률       0
# 거래량       0
# 금액(백만)    0
# 신용비       0
# 개인        0
# 기관        0
# 외인(수량)    0
# 외국계       0
# 프로그램      0
# 외인비       0
# dtype: int64
# """


# # # 50으로 나누고 곱함
# datasets.iloc[662:,0:4] = datasets.iloc[662:,0:4]/50.0
# datasets.iloc[662:,5] = datasets.iloc[662:,5]*50
# datasets = datasets.iloc[::-1,:]

# # 빈자리에 임의이 값 추가.
# datasets.iloc[-1:,4] = 0.8
# datasets.iloc[-1,7]  = datasets.iloc[-2,7]
# datasets.iloc[-1,10] = datasets.iloc[-2,10]
# """
# 그래프 확인한 뒤에 필요한 열에 타입변환 및 순서 바꾸기.@@@@@@@@@@@@@@@@@@
# """
# """
# 0~13
# del datasets["시가"]
# del datasets["고가"]
# del datasets["저가"]
# del datasets["종가"]
# del datasets["등락률"]
# del datasets["거래량"]
# del datasets["금액(백만)"]
# del datasets["신용비"]
# del datasets["개인"]
# del datasets["기관"]
# del datasets["외인(수량)"]
# del datasets["외국계"]
# del datasets1["프로그램"]
# del datasets1["외인비"]
# """
# # 안쓸 컬럼 제거
# del datasets["거래량"]
# del datasets["등락률"]
# del datasets["기관"]
# del datasets["외인(수량)"]
# del datasets["외국계"]
# del datasets1["프로그램"]

# # 저장
# datasets.to_csv('../data/csv/csv.csv', encoding='ms949', sep=",")


# # 불러오기 ===============================================================================================
# datasets = pd.read_csv('../data/csv/csv.csv',index_col=0 ,encoding='ms949')

# # 상관계수 및 컬럼 확인
# print(datasets.corr())
# import matplotlib.pyplot as plt
# sns.set(font_scale=0.7)
# sns.heatmap(data=datasets.corr(), square=True, annot=True, cbar=True)
# plt.show()

# 몇일 분량, 컬럼확인
size=5
col=8

# # y데이터 생성
# y = datasets.iloc[size-1:,3].values #(2378,)
# x_prd = datasets.iloc[-size-1:-1,:col].values
# print(x_prd.shape)
# """
# 그래프 확인하기. y 값 종가 컬럼으로 바꾸기.@@@@@@@@@@@@@@@@@@

# 상관계수 시각화
# print(datasets.corr())
# import matplotlib.pyplot as plt
# sns.set(font_scale=0.7)
# sns.heatmap(data=datasets.corr(), square=True, annot=True, cbar=True)
# plt.show()
# """
# # 전처리 ===================================================================================

# # #MinMax
# scaler = MinMaxScaler()
# scaler.fit(datasets)
# datasets_minmaxed = scaler.fit_transform(datasets)

# # train_test_split
# x = split_x(datasets_minmaxed,size,col) # (2378,20,9)

# # 넘파이 저장
# np.save('../data/npy/samsung_2.npy',arr=([x,y,x_prd]))
x = np.load('../data/npy/samsung_2.npy',allow_pickle=True)[0]
y = np.load('../data/npy//samsung_2.npy',allow_pickle=True)[1]
x_prd = np.load('../data/npy//samsung_2.npy',allow_pickle=True)[2]

x_prdD = pd.DataFrame(x_prd)
x_prdD.to_csv('../data/csv/x_prdD.csv', encoding='ms949', sep=",")

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, shuffle=True)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2 , shuffle=True)

x_train=x_train.reshape(-1,size,col).astype('float32')
x_test=x_test.reshape(-1,size,col).astype('float32')
x_val=x_val.reshape(-1,size,col).astype('float32')
x_prd = x_prd.reshape(-1,size,col).astype('float32')


# 모델===============================================================================

model = Sequential()

# model.add(LSTM(128,input_shape=(x_train.shape[1],x_train.shape[2])))
# # model.add(Dense(512,activation='relu'))
# # model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(1, activation='relu'))
# model.summary()

# 콜백 ===============================================================================
modelpath = "../data/h5/Samsung_aest_model.hdf5"
es = EarlyStopping(monitor = 'val_loss',patience=100, mode="min")
cp = ModelCheckpoint(monitor = 'val_loss',filepath = modelpath, save_best_only=True)#, mode='auto')

model = load_model('../data/h5/Samsung_aest_model.hdf5')
# model.compile(loss = 'mse',optimizer = 'adam')
hist = model.fit(x_train,y_train, validation_data=(x_val,y_val),epochs=10000,batch_size=64,verbose=2,callbacks=[es,cp], shuffle=True)

# 로스 시각화
plt.rc('font', family='Malgun Gothic')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('loss ')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train_loss','val_loss'])
plt.show()

model.save('../data/h5/samsung_maw.h5')
model.save_weights('../data/h5/samsung_maw_weight.h5')
# model = load_model('../data/h5/samsung_maw.h5')

from sklearn.metrics import r2_score

# model = load_model('../data/h5/Samsung_best_model.hdf5')
# loss = model.evaluate(x_test,y_test,batch_size=8)
# print("loss : ",loss )

# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)


# model = load_model('../data/h5/Samsung_aest_model.hdf5')
loss = model.evaluate(x_test,y_test, batch_size= 64)
print("loss, mse : ",loss )

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

y_predict2 = model.predict(x_prd)
print("예상 값",y_predict2)


plt.figure(figsize=(9,3.7))

plt.subplot(2,1,1)
plt.plot(y_predict, label='y_predict')
plt.legend()

plt.subplot(2,1,2)
plt.plot(y_test, label='y_test')
plt.legend()
plt.show()

# csv만들기
y_test = pd.DataFrame(y_test)
y_test['Target'] = y_predict
y_test.to_csv('../data/csv/y_test.csv', encoding='ms949', sep=",")

"""
loss :  2903743.25
R2 :  0.9829335258378604
예상 값 [[22940.273]]

loss :  3025.900634765625
R2 :  0.9999820973211386
예상 값 [[25403.98]]

loss :  58348.70703125
R2 :  0.9996583404017011
예상 값 [[47738.395]]

loss :  2139.6181640625
R2 :  0.9999878286776926
예상 값 [[23881.008]]

loss :  3025.900634765625
R2 :  0.9999820973211386
예상 값 [[25403.98]]

loss :  2685.16015625
R2 :  0.999983363154749
예상 값 [[23503.31]]

loss :  1340.1365966796875
R2 :  0.9999925828902324

loss :  6652.8037109375
R2 :  0.9999601217251832
예상 값 [[45357.98]]

loss :  2037.7379150390625
R2 :  0.9999878561476985
예상 값 [[23678.729]]

loss :  1308.367431640625
R2 :  0.9999917872351053
예상 값 [[47014.367]]

loss :  2394.91748046875
R2 :  0.9999855578125276
예상 값 [[26249.912]]

loss, mse :  [20249.974609375, 20249.974609375]
R2 :  0.9998772207308237
예상 값 [[51183.676]]

loss, mse :  [10652.0751953125, 10652.0751953125]
R2 :  0.9999376085010482
예상 값 [[17508.754]]

loss, mse :  [16550.001953125, 90.79109191894531]
R2 :  0.9999026399203765
예상 값 [[59964.156]]

# column : 4 batch 3
loss, mse :  [121269.078125, 223.651611328125]
R2 :  0.9993123751092797
예상 값 [[182986.03]]

50일치
loss, mse :  5751.970703125
R2 :  0.9999271175883038
예상 값 [[1106.4974]]
 왜인지 예상 값이 현실성 없음.. 뭐가 틀렸는지 모르겠음. 

5일치
loss, mse :  676060.75
R2 :  0.9913746023231675
예상 값 [[1961.6686]]
 왜인지 예상 값이 현실성 없음.. 뭐가 틀렸는지 모르겠음. 

loss, mse :  677371.25
R2 :  0.9913578814153434
예상 값 [[32445.424]]

핏과 트레인 테스트 쉐이프가 shuffle=True 일 때 8만이상이 나오긴함.
근데, 신뢰가 안감
loss, mse :  2860.739990234375
R2 :  0.9999846045480831
예상 값 [[92604.54]]

loss, mse :  2637.56201171875
R2 :  0.9999857230678582
예상 값 [[83970.1]]

loss, mse :  3937.543212890625
R2 :  0.9999778255667975
예상 값 [[84239.6]]
"""