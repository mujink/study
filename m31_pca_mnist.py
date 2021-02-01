import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

print(x.shape)      # (70000, 28, 28)

# 실습 
# pca를 통해 0.95 이상인거 몇개?
# pca 배운거 다 집어 넣고 확인!!

x = x.reshape(-1,28*28)
x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

# 컬럼을 n_components 수 만큼 압축한다
# pca = PCA(n_components=9)
pca = PCA()
x2 = pca.fit_transform(x)
print(x2.shape)                 # (70000, 784)

# 컬럼을 압축한 컬럼의 변화 비율을 확인
# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))             # 1.000000000000002

pca = PCA(154)
pca.fit(x_train)
pca.fit(x_test)
# pca.fit(x_test)
# cumsum = 누적합계
cumsum = np.cumsum(pca.explained_variance_ratio_)
print("누계 :", cumsum)
d = np.argmax(cumsum >= 0.95)+1
# print(np.argmax(cumsum >= 0.95))
print("cumsum >= 0.95", cumsum >= 0.95)
print("d :", d)
"""
합계 : 1.000000000000002

누적 합계 : [0.09746116 0.16901561 0.23051091 0.28454476 0.3334341  0.37648637
 0.40926898 0.4381654  0.46574904 0.48917044 0.51023733 0.53061286
 0.5476835  0.5646237  0.58045752 0.59532097 0.60851456 0.6213047
 0.63317742 0.64470679 0.65536719 0.66546513 0.67505665 0.684153
 0.69298586 0.70137405 0.70947236 0.71732954 0.72473217 0.73163231
 0.73819375 0.74464845 0.75065664 0.75651276 0.7621803  0.767615
 0.77266217 0.77753297 0.78232252 0.78699846 0.79154214 0.79599132
 0.80017349 0.80413513 0.8079722  0.81173005 0.81534432 0.81883456
 0.82222188 0.82541884 0.82858738 0.83168883 0.83465363 0.83752465
 0.84034978 0.84304401 0.84572793 0.84829303 0.85082471 0.85327119
 0.85566821 0.85805402 0.86034635 0.86255584 0.86468645 0.86674962
 0.86877744 0.87072778 0.87264249 0.87452799 0.87639774 0.87819878
 0.87996665 0.88170024 0.88334873 0.88498109 0.88659517 0.8881382
 0.88960839 0.89103038 0.89244053 0.89384198 0.89523801 0.89658823
 0.89791193 0.89923082 0.90052278 0.90177448 0.90299975 0.90420391
 0.90536751 0.90651066 0.90763609 0.90873461 0.90981796 0.91088955
 0.91192631 0.91296042 0.91396596 0.91496505 0.91594165 0.91688281
 0.91781835 0.91872968 0.91962998 0.92051924 0.92138052 0.92223248
 0.92307238 0.92388779 0.92467296 0.92544914 0.92622349 0.92698707
 0.92774686 0.92849441 0.92922401 0.92994821 0.9306625  0.9313652
 0.93205437 0.932739   0.93341583 0.93408575 0.93474494 0.93538457
 0.93601355 0.936637   0.93725268 0.93785055 0.93844261 0.93903264
 0.93961348 0.94019281 0.94076751 0.94133335 0.94188548 0.94243136
 0.94296058 0.9434779  0.94398849 0.9444881  0.94498075 0.94546892
 0.94595499 0.94643158 0.9469077  0.94737693 0.94783557 0.94829104
 0.9487428  0.94918293 0.94961448 0.95003523]

cumsum >= 0.95 [False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False False False False  True]
d : 154


"""

# 1 data
# x_train, x_test, y_train, y_test = train_test_split(
#     x2, y, train_size=0.8, random_state=44
# )
# 2 model
model = RandomForestClassifier()

# 3 fit
model.fit(x_train, y_train)

# 4 evel
acc = model.score(x_test, y_test)

# print(model.feature_importances_)
print("acc :", acc)

"""
pca = PCA(154)

acc : 0.9695
"""