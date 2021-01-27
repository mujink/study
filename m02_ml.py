#  argmax 사용  y_predict 최대값 출력
#  sklearn.onehotencoding
import numpy as np
from sklearn.datasets import load_iris

datasets = load_iris()
x = datasets.data
y = datasets.target
print(datasets.DESCR)
print(datasets.feature_names)
print(datasets.target_names)


"""
 :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988
"""
print(x.shape)      #(150,4)
print(y.shape)      #(150,3)

#1.1 Data Preprocessing / train_test_splitm /  MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
"""
#  sklearn.onehotencoding
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           #. oneHotEncoder load
y = y.reshape(-1,1)                 #. y_train => 2D
one.fit(y)                                #. Set
y = one.transform(y).toarray()      #. transform
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=1)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)



#2.model
from sklearn.svm import LinearSVC


#  머신러닝 모델 ===================
# 머신런닝 모델을 한줄로 끝남
# 핏은 동일
model = LinearSVC()
model.fit(x,y)
# 스코어와 프래딕트 평가 예측
result = model.score(x,y)
print(result)

y_pred = model.predict(x_train[-5:-1])
print(y_pred)
print(y_train[-5:-1])

"""
loss :  0.039841461926698685
acc :  0.9666666388511658
"""