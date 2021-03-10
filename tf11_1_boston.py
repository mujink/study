# 회귀 맹그러
# sklearn에 R2 스코어 쓸 것

from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.metrics import r2_score
tf.set_random_seed(66)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

dataset = load_boston()

x_data = dataset.data
y_data = dataset.target.reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

fd = {x:x_data ,y:y_data}

w = tf.Variable(tf.random_normal([13,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis =  tf.matmul(x, w) + b



cost = tf.compat.v1.reduce_mean(tf.square(hypothesis-y))
# train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=0.1).minimize(cost)
# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
import numpy as np
with tf.compat.v1.Session() as sess :
    sess.run(tf.compat.v1.global_variables_initializer())
        
    for step in range(10000):
        _, cost_val, hy_val = sess.run([train, cost, hypothesis], feed_dict=fd)
        if step % 20 ==0:
            # print(step,"cost :", cost_val, "\n hy :",hy_val)
            print(step,"cost :", cost_val)#, "\n hy :",hy_val)
    h = sess.run([hypothesis], feed_dict=fd)
    h = np.array(h).reshape(-1, 1)
    r2_m1 = r2_score(y_data, h)
    print("R2 :", r2_m1)

# R2 : 0.7397640620684738