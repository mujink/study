import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
def relu(x):
    return np.maximum(0,x)
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

x = np.arange(-5, 5, 0.1)
y = mish(x)

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()


# 과제
# elu selu, reaky relu
# 그리기