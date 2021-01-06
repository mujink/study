


import numpy as np

datasets = np.array(range(1,11)) #1~10
size = 5

# def split_x(seq, size):
#     aaa=[]                                          # aaa는 list임
#     for i in range(len(seq)-size+1):                # for 0에서 seq 변수의 어레이의 길이에서 사이즈의 값을 뺀 만큼 다음을 반복
#         # print(i)
#         subset = seq[i : (i+size)]                  # subset = seq 어레이 [i:(i+size)] 범위의 리스로 초기화함. 
#         aaa.append([item for item in subset])       # aaa라는 리스트에 다음 초기화된 [subset]를 값을 맴버로 추가함
#     # print(type(aaa))
#     # print(subset)
#     return np.array(aaa)                            # np.array를 aaa[] 리스트로 초기와 후 반환함.

def split_x(seq, size):
    aaa=[]                                          # aaa는 list임
    for i in range(len(seq)-size+1):                # for 0에서 seq 변수의 어레이의 길이에서 사이즈의 값을 뺀 만큼 다음을 반복
        subset = seq[i : (i+size)]                  # subset = seq 어레이 [i:(i+size)] 범위의 리스로 초기화함. 
        aaa.append(subset)                          # aaa라는 리스트에 다음 초기화된 [subset]를 값을 맴버로 추가함
    print(type(aaa))
    return np.array(aaa)                            # np.array를 aaa[] 리스트로 초기와 후 반환함.



print(split_x(datasets,size))
# dataset = split_x(a, size)
# print("==========================")
# print(dataset)
