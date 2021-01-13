import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
"""
.DataFrame : 펜다스의 다차원 배열

.columns 
.index              줄 37~39
.head()             줄 119~127
.tail()

.shape
.info()
.describe()
.isnull()
.isnull().sum()

.corr()
.value_counts()

"""
dataset = load_iris()
# print(dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
# print(dataset.values())

x = dataset.data
# y = dataset['target']
y = dataset.target          # dat
print(x.shape, y.shape) # (150, 4) (150,)
print(type(x), type(y)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>


# df = pd.DataFrame(x, columns=dataset['feature_names'])
df = pd.DataFrame(x, columns=dataset.feature_names)
"""
 인덱스 기본값 =  RangeIndex(start=0, stop=데이터길이, step=1)
 columne : header
    Index(['columne'], dtype='object')
# """ 
print(df)
"""
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                  5.1               3.5                1.4               0.2
1                  4.9               3.0                1.4               0.2
2                  4.7               3.2                1.3               0.2
3                  4.6               3.1                1.5               0.2
4                  5.0               3.6                1.4               0.2
..                 ...               ...                ...               ...
145                6.7               3.0                5.2               2.3
146                6.3               2.5                5.0               1.9
147                6.5               3.0                5.2               2.0
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8

[150 rows x 4 columns]
"""
print(df.shape)
"""
(150, 4)
"""
print(df.columns)
# Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
#        'petal width (cm)'],
#       dtype='object')

print(df.index) # RangeIndex(start=0, stop=150, step=1)

print(df.head()) #df[:5]
print(df.tail()) #df[-5:]
print(df.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
dtypes: float64(4)
memory usage: 4.8 KB
None
# """
print(df.describe())
"""
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
count         150.000000        150.000000         150.000000        150.000000
mean            5.843333          3.057333           3.758000          1.199333
std             0.828066          0.435866           1.765298          0.762238
min             4.300000          2.000000           1.000000          0.100000
25%             5.100000          2.800000           1.600000          0.300000
50%             5.800000          3.000000           4.350000          1.300000
75%             6.400000          3.300000           5.100000          1.800000
max             7.900000          4.400000           6.900000          2.500000
"""

print(df.columns)
# df.columns = ['sepal_length','sepal_width','petal_length','petal_width']
print(df.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
dtypes: float64(4)
memory usage: 4.8 KB
None
"""
#  y 컬럼 추가하기.

df['Target'] = dataset.target
print(df.head())
"""
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0
2                4.7               3.2                1.3               0.2       0
3                4.6               3.1                1.5               0.2       0
4                5.0               3.6                1.4               0.2       0
"""

print(df.shape) # (150, 5)
print(df.columns)
# # Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Target'], dtype='object')
print(df.index)
# RangeIndex(start=0, stop=150, step=1)
print(df.tail())
"""
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target
145                6.7               3.0                5.2               2.3       2
146                6.3               2.5                5.0               1.9       2
147                6.5               3.0                5.2               2.0       2
148                6.2               3.4                5.4               2.3       2
149                5.9               3.0                5.1               1.8       2
"""

print(df.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
 4   Target             150 non-null    int32
dtypes: float64(4), int32(1)
memory usage: 5.4 KB
None
""" 
#  널값 찾기
print(df.isnull())
"""
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target
0                False             False              False             False   False
1                False             False              False             False   False
2                False             False              False             False   False
3                False             False              False             False   False
4                False             False              False             False   False
..                 ...               ...                ...               ...     ...
145              False             False              False             False   False
146              False             False              False             False   False
147              False             False              False             False   False
148              False             False              False             False   False
149              False             False              False             False   False

[150 rows x 5 columns]
"""
#  널값 Summary
print(df.isnull().sum())
# sepal length (cm)    0
# sepal width (cm)     0
# petal length (cm)    0
# petal width (cm)     0
# Target               0
# dtype: int64
print(df.describe())
"""
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)      Target
count         150.000000        150.000000         150.000000        150.000000  150.000000
mean            5.843333          3.057333           3.758000          1.199333    1.000000
std             0.828066          0.435866           1.765298          0.762238    0.819232
min             4.300000          2.000000           1.000000          0.100000    0.000000
25%             5.100000          2.800000           1.600000          0.300000    0.000000
50%             5.800000          3.000000           4.350000          1.300000    1.000000
75%             6.400000          3.300000           5.100000          1.800000    2.000000
max             7.900000          4.400000           6.900000          2.500000    2.000000
"""
#  타겟 벨류값 카운팅
print(df['Target'].value_counts())
"""
2    50
1    50
0    50
"""
# Correlation  상관계수 히트맵
print(df.corr())
"""
Name: Target, dtype: int64
                   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    Target
sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
Target                      0.782561         -0.426658           0.949035          0.956547  1.000000
"""
# 상관계수 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set(font_scale=1.2)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()

#  도수 분포표. 측정 값으 분포도 표시 (가로축은 값, 세로 축은 가로축 값에 대한 분포)
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.hist(x = 'sepal_length', data=df)
# hist : 
#       data : 데이터 프레임.
#       x : 데이터 프레임의 column.

plt.title('sepal_length')

plt.subplot(2,2,2)
plt.hist(x = 'sepal_width', data=df)
plt.title('sepal_width')

plt.subplot(2,2,3)
plt.hist(x = 'petal_length', data=df)
plt.title('petal_length')

plt.subplot(2,2,4)
plt.hist(x = 'petal_width', data=df)
plt.title('petal_width')
plt.show()