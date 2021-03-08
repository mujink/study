from sklearn.covariance import EllipticEnvelope
import numpy as np

# aaa = np.array([[1,2,-10000,3,4,6,7,8,90,100,5000]])
aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
                [10000,20000,3,40000,50000,60000,70000,8,90000,100000]])

# aaa = np.array([[10000,20000,3,40000,50000,60000,70000,8,90000,100000],
#                 [1,2,3,4,10000,6,7,5000,90,100]])

aaa = np.transpose(aaa)

print(aaa.shape)
# 가우시안 분포 공분산 뭐시기
outlier = EllipticEnvelope(contamination=.2)
outlier.fit(aaa)

print(outlier.predict(aaa))