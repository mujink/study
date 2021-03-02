import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
def sigmoid (x):
    return 1 / ( 1+ np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
def softmax(x):
    return np.exp(x)/ np.sum(np.exp(x))
def elu(x,alpha):
    return (x>0)*x+(x<=0)*(alpha*(np.exp(x)-1))
def Leaky_relu(x):
    return np.maximum(0.01*x,x)
def selu(x,scale,alpha):
    return scale*elu(x,alpha)


x = np.arange(-5, 5, 0.1)
y = selu(x, 1.5,1.5)
# y = np.tanh(x)

ratio = y
labels = y

# plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.plot(x,y)
plt.show()


# 과제
# elu, selu, reaky relu
# 그리기