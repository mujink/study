# 데이터 크기를 맞춘뒤 저장한 CSV 파일을 로드 => 컬럼을 늘리지 않는 한, 다시 확인할 필요 없음.
# 데이터와 타겟 사이 상관계수 최종 확인. => 완료시 중간저장한 CSV 파일은 건드리지 않음.
# 전처리 미스로 로스값이 높음 => minmax 이전에 x 값, y 값 중간확인 필요.
# 확인은 CSV 파일이 편하므로 x,y 스플릿 후 각각 CSV 파일로 저장하는 작업을 실행함.
# 각각 작업은 CSV 파일로 변환하여 확인하고 주석처리로 하나하나 확인함.

# Split 함수에 Transpose 적용한 버전임. 수업시간에 시계열에대한 설명을 행을 고정한체 에서 열을 옮겨가며 설명하는 것을 떠올려 수정해봄.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 함수정의
# len(Seq) : 데이터 셋의 행의 길이 현재 행의 길이는 1085임
# size : 몇 일치 분량 일지         현재 전날 5일 분량으로 다음날을 확인하고자 함.
# col : 몇 일치 분량의 열인지      삼성 5, 코스닥 6 컬럼의 5일치를 확인하고자 함.
def split_x(seq,size,col):
    import numpy as np
    print(range(len(seq)-size-1))
    print(seq.shape)
    aaa=[]
    for i in range(len(seq)-size-1):                     
        subset = seq[i:(i+size),0:col].astype('float32') 
        subset = np.transpose(subset)
        aaa.append(np.array(subset))
    print(np.array(aaa).shape)
    return np.array(aaa)

def split_y(seq,size,col):
    import numpy as np
    print(range(len(seq)-size-1))
    print(seq.shape)
    aaa=[]
    for i in range(len(seq)-size-1):
        subset = seq[(i+size),col].astype('float32')
        subset1 = seq[(i+size+1),col].astype('float32')
        aaa.append(np.array([subset,subset1]))
        # aaa = np.transpose(aaa)
        # aaa.append(np.array())
    print(np.array(aaa).shape)
    return np.array(aaa)
# # 불러오기 ===============================================================================================

datasets = pd.read_csv('./csv/sam_dc.csv',index_col=0 ,encoding='ms949')
codacdbset = pd.read_csv('./csv/codac150_dc.csv',index_col=0, encoding='ms949')

# 두 데이터 타겟간 상관 계수 비교===============================================================================
#  왼쪽 위가 코스닥 오른쪽 아래가 삼성임.
# sd = codacdbset
# sd['삼성 @@'] = datasets["시가"]
# sd['삼성 고가'] = datasets["고가"]
# # sd['삼성 저가'] = datasets["저가"]
# # sd['삼성 종가'] = datasets["종가"]
# # sd['삼성 등락률'] = datasets["등락률"]
# sd['삼성 거래량'] = datasets["거래량"]
# sd['삼성 금액(백만)'] = datasets["금액(백만)"]
# # sd['삼성 신용비'] = datasets["신용비"]
# sd['삼성 개인'] = datasets["개인"]
# # sd['삼성 기관'] = datasets["기관"]
# # sd['삼성 외인(수량)'] = datasets["외인(수량)"]
# # sd['삼성 외국계'] = datasets["외국계"]
# sd['삼성 프로그램'] = datasets["프로그램"]
# # sd['삼성 ##'] = datasets["외인비"]
# print(sd.shape)
# # 상관계수 및 컬럼 확인
# print(sd.corr())
# sns.set(font_scale=0.7)
# sns.heatmap(data=sd.corr(), square=True, annot=True, cbar=True)
# plt.show()
#========================================================================================================

# 몇일 분량, 컬럼확인
# 컬럼과 사이즈는 데이터 컬럼, 데이터의 수량 보다 많을 수 없음.
size=5
sam_col=6
cdc_col=5

datasets.to_numpy()
codacdbset.to_numpy()
# x1은 삼성, x2는 코스닥, y는 삼성.
x1 = datasets.to_numpy()
x2 = codacdbset.to_numpy()
y = datasets.to_numpy()


# x y데이터 생성, 스플릿
x1 = split_x(x1,size,sam_col)
x2 = split_x(x2,size,cdc_col)
y = split_y(y,size,0)


# 샘플 저장 및 확인=================================================================================================
# 넘파이는 CSV 저장이 안되서 데이터 프레임을 씌움, 2D 보다 높은 차원은 데이터프레임이 적용되지 않음.
# 출력 이상없음.
# +++++++++++++++++++++++++++++++++++
# x1 = (1079, 5, 6).astype('float32') 
# x2 = (1079, 5, 5).astype('float32') 
# y  = (1079, 2).astype('float32') 
# x1_prd.shape = (1,5,6)
# x2_prd.shape = (1,5,6)
# +++++++++++++++++++++++++++++++++++
# x1 = pd.DataFrame(x1[-1])
# x2 = pd.DataFrame(x2[-1])
# y = pd.DataFrame(y)
# 샘플 데이터는 확인용 데이터 이므로 로드하여 사용하지 않음.
# x1.to_csv('./csv/x1.csv', encoding='ms949', sep=",")
# x2.to_csv('./csv/x2.csv', encoding='ms949', sep=",")
# y.to_csv('./csv/y.csv', encoding='ms949', sep=",")
# 샘플 확인됨=================================================================================================

# MinMax=====================================================================================================
# 적용 시점을 조정할 필요가 있음. 적용시점에 따라 로스 출력이 크게 변할 수 있음.
# 데이터 스프릿  전에 적용하면 스플릿 이후 데이터 값이 의도대로 적용 되었는지 확인하기 어려움
# 그러나 스플릿 이후에 적용 하여 프린트 해보면, 데이터의 어떤 인덱스를 선택하든 항상 0과 1이 출력되는 행렬을 볼 수 있음.
# 인덱스마다 항상 0과 1이 출력되는 데이터가 loss 출력을 높이게됨.

# scaler = MinMaxScaler()
# scaler.fit(x1)
# x1 = scaler.fit_transform(x1)

# scaler.fit(x2)
# x2 = scaler.fit_transform(x2)
# MinMax=====================================================================================================

# x_Predict =================================================================================================
#  생성 삼성, 코스닥 등 각 마지막 순번의 행렬로 초기화
# train_test_split이 model.fit을 위해 shuffle할 예정이므로 이전에 초기화 해야함.

x1_prd = np.array(x1[-1:]) 
x2_prd = np.array(x2[-1:])
print(x1_prd.shape)     #(1,5,6)
print(x2_prd.shape)     #(1,5,5)


# train_test_split=========================================================================================
# 입력 데이터의 길이가 맞는지 확인
print("X1 :", x1.shape)     #(1079,5,6)
print("X2 :", x2.shape)     #(1079,5,5)
print("Y:", y.shape)        #(1079,2)

x1_train,x1_test,x2_train,x2_test,y_train,y_test = train_test_split(x1, x2, y,test_size=0.2, shuffle=True, random_state=3)
x1_train,x1_val,x2_train,x2_val,y_train,y_val = train_test_split(x1_train,x2_train,y_train, test_size=0.2 , shuffle=True)


# 리세이프==================================================================================================
# x1_train=x1_train.reshape(-1,size,sam_col)
# x2_train=x2_train.reshape(-1,size,cdc_col)
# x1_test=x1_test.reshape(-1,size,sam_col)
# x2_test=x2_test.reshape(-1,size,cdc_col)
# x1_val=x1_val.reshape(-1,size,sam_col)
# x2_val=x2_val.reshape(-1,size,cdc_col)
# x1_prd = x1_prd.reshape(-1,size,sam_col)
# x2_prd = x2_prd.reshape(-1,size,cdc_col)

# 최종 리세이프 확인
print("X1 Train :", x1_train.shape,"X2 Trian :", x2_train.shape)
print("Y1 Train :", y_train.shape, type(x1_train), type(x2_train), type(y_train))
print("X1 Tset  :", x1_test.shape," X2 Test  :", x2_test.shape)
print("Y1 Tset  :", y_test.shape, type(x1_test), type(x2_test), type(y_test))
print("X1 Val   :", x1_val.shape,"  X2 val   :", x2_val.shape)
print("Y1 Val   :", y_val.shape, type(x1_val), type(x2_val), type(y_val))
print("X1_prd   :", x1_prd.shape,"  X2_prd   :", x2_prd.shape)
print(type(x1_prd), type(x2_prd))

# # #  넘파이 저장=======================================================

np.save('./npy/1.npy',x1_train)
np.save('./npy/2.npy',x2_train)
np.save('./npy/3.npy',y_train)
np.save('./npy/11.npy',x1_test)
np.save('./npy/12.npy',x2_test)
np.save('./npy/13.npy',y_test)
np.save('./npy/21.npy',x1_val)
np.save('./npy/22.npy',x2_val)
np.save('./npy/23.npy',y_val)
np.save('./npy/31.npy',x1_prd)
np.save('./npy/32.npy',x2_prd)

# # #  넘파이 불러오기=======================================================


# x1_train = np.load('./npy/1.npy',allow_pickle=True)
# x2_train = np.load('./npy/2.npy',allow_pickle=True)
# y_train = np.load('./npy/3.npy',allow_pickle=True)
# x1_test = np.load('./npy/11.npy',allow_pickle=True)
# x2_test = np.load('./npy/12.npy',allow_pickle=True)
# y_test = np.load('./npy/13.npy',allow_pickle=True)
# x1_val = np.load('./npy/21.npy',allow_pickle=True)
# x2_val = np.load('./npy/22.npy',allow_pickle=True)
# y_val = np.load('./npy/23.npy',allow_pickle=True)
# x1_prd = np.load('./npy/31.npy',allow_pickle=True)
# x2_prd = np.load('./npy/32.npy',allow_pickle=True)
