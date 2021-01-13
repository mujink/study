import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header = 0)
"""
index_col = 0 : 인덱스 0 은 인덱스가 아님.
header = 0 : 인덱스 0은 헤더임.
"""
print(df)
print(type(df))


print(df.shape) # (150, 5)
print(df.info())
"""
<class 'pandas.core.frame.DataFrame'>
Int64Index: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
 4   Target             150 non-null    int64
dtypes: float64(4), int64(1)
"""

#  53 분까지, 판다스를 넘파이로 바꾸는 것을 찾아랏!!!
""" https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_numpy.html

DataFrame.to_numpy(dtype=None, copy=False, na_value=<object object>)[source]
Convert the DataFrame to a NumPy array.

By default, the dtype of the returned array will be the common NumPy dtype of all types in the DataFrame. For example, if the dtypes are float16 and float32, the results dtype will be float32. This may require copying data and coercing values, which may be expensive.

Parameters
dtypestr or numpy.dtype, optional
The dtype to pass to numpy.asarray().

copybool, default False
Whether to ensure that the returned value is not a view on another array. Note that copy=False does not ensure that to_numpy() is no-copy. Rather, copy=True ensure that a copy is made, even if not strictly necessary.

na_valueAny, optional
The value to use for missing values. The default value depends on dtype and the dtypes of the DataFrame columns.

New in version 1.1.0.

Returns
numpy.ndarray

Examples

pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_numpy()
array([[1, 3],
       [2, 4]])
With heterogeneous data, the lowest common type will have to be used.

df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.5]})
df.to_numpy()
array([[1. , 3. ],
       [2. , 4.5]])
# """

aaa = df.to_numpy()
print(aaa)
print(type(aaa))
#  or
bbb = df.values
print(bbb)
print(type(bbb))

np.save('../data/npy/iris_sklearn.npy', arr=(aaa)) 

# 과제
# 판다스의 loc, iloc에 대해 정리

# loc : 행열을 키문자 기준으로 추출함.
# iloc : 행열을 인덱스 기준으로 추출함.

# row:
df.iloc[0] # data의 첫번째 행만
df.iloc[1] # 두번째 행만
df.iloc[-1] # 마지막 행만
# Columns:
df.iloc[:,0] # 첫번째 열만
df.iloc[:,1] # 두번째 열만
df.iloc[:,-1] # 마지막 열만


df.iloc[0:5] # 첫 5개행만
df.iloc[:, 0:2] # 첫 2개열만
df.iloc[[0,3,6,24], [0,5,6]] # 1st, 4th, 7th, 25th 행과 + 1st 6th 7th 열만
df.iloc[0:5, 5:8] # 첫 5개 행과 5th, 6th, 7th 열만

# ========================================================
df.loc['Andrade']  # Andrade 행만 선택
df.loc[['Andrade','Veness']] # Andrade와 Veness 둘다 선택

df.loc[['Andreade','Veness'],['first_name', 'address', 'city']] # .loc[[행의 키문자],[열의 키문자]]