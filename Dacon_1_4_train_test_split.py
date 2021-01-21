import numpy as np



#  넘파이 불러오기=======================================================

x = np.load('../data/csv/Dacon/np/TrainDb_X.npy',allow_pickle=True)
y = np.load('../data/csv/Dacon/np/TrainDb_Y.npy',allow_pickle=True)


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




# train_test_split ==========================================================================================
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = False)#, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = False)#, random_state=1)
print("# shape  test=====================================================================================================")
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)

print(x_train)
print(y_train)

# reshape ===================================================================================================
