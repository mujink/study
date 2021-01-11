# cnn으로 구성
# 2차원을 4차원으로 늘여서 하시오/

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)             # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)               # (10000, 28, 28) (10000,)

print(x_train[0])
print(x_train[0].shape)
print("y_train[0] :",y_train[0])

# plt.imshow(x_train[0], 'gray')
plt.imshow(x_train[0])
plt.show()




# import matplotlib.pyplot as plt
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['acc'])
# plt.title('cnn iris')
# plt.ylabel('loss, acc')
# plt.xlabel('epoch')
# plt.legend(['train loss', 'acc'])
# plt.show()

