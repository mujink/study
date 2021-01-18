import numpy as np

"""
split_xy 함수
다:다, 1:1, 범용 하지만 출력 값을 일일히 확인해야함.
"""

def split_xy(seq,x_size,x_col_start, x_col_end ,y_size,y_col_start,y_col_end):
    print(range(len(seq)-x_size-1))                             
    print(seq.shape)                                            
    x=[]
    y=[]
    for i in range(len(seq)-x_size-y_size+1):                          
        xi = seq[i:(i+x_size),x_col_start-1:x_col_end].astype('float32')   
        yi = seq[(i+x_size):(i+x_size+y_size),y_col_start-1:y_col_end].astype('float32')       
        x.append(np.array(xi))          
        y.append(np.array(yi))          
    print(np.array(x).shape)
    print(np.array(y).shape)
    return np.array(x),  np.array(y)

# # 불러오기 ===============================================================================================

# datasets = pd.read_csv('../data/csv/sam_dc.csv',index_col=0 ,encoding='ms949')

# len(Seq) : 데이터 셋의 행의 길이 현재 행의 길이는 1085임
# x_size : 몇 일치 분량 일지         현재 전날 5일 분량으로 다음날을 확인하고자 함.
# x_col : 몇 일치 분량의 열인지      삼성 5, 코스닥 6 컬럼의 5일치를 확인하고자 함.
# y_size : 몇 일치 분량 일지         현재 전날 5일 분량으로 다음날을 확인하고자 함.
# x_col : 몇 일치 분량의 열인지      삼성 5, 코스닥 6 컬럼의 5일치를 확인하고자 함.
#=========================================================================================================

dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                  [11,12,13,14,15,16,17,18,19,20],
                  [21,22,23,24,25,26,27,28,29,30],
                  [31,32,33,34,35,36,37,38,39,40],
                  [41,42,43,44,45,46,47,48,49,50]])
                  
dataset = dataset.transpose()
"""(10,5)
[[ 1 11 21 31 41]
 [ 2 12 22 32 42]
 [ 3 13 23 33 43]
 [ 4 14 24 34 44]
 [ 5 15 25 35 45]
 [ 6 16 26 36 46]
 [ 7 17 27 37 47]
 [ 8 18 28 38 48]
 [ 9 19 29 39 49]
 [10 20 30 40 50]]
"""
print(dataset)
# 입력은 여기에 =========
x_size = 3
x_col_start = 3
x_col_end = 5
y_size = 1
y_col_start = 1
y_col_end = 2
# 입력은 여기 =========

# x y데이터 생성, 스플릿
xset, yset = split_xy(dataset,x_size,x_col_start, x_col_end ,y_size,y_col_start,y_col_end)
# 입력 데이터의 길이가 맞는지 확인
print(xset)
print(yset)
