import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float)

# 7분이먼 충분하겠지 맹그러라.


x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

fd ={x:x_data,y:y_data}

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,2]), name="weight")
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name="weight")
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([2]), name="bias")
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name="bias")

hypothesis = tf.nn.relu(tf.matmul(x, w) + b)
hypothesis1 = tf.sigmoid(tf.matmul(hypothesis, w) + b1)
# [실습] 맹그러봐
cost = -tf.reduce_mean(y*tf.log(hypothesis1)+(1-y)*tf.log(1-hypothesis1))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis1 > 0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(15001):
        cost_val, _ = sess.run([cost, train], feed_dict=fd)
        if step % 200 ==0:
            print(step,"cost :", cost_val)
    
    h, c, a = sess.run([hypothesis1, predicted, accuracy], feed_dict=fd)
    print("예측 값 :", h, "\n 원래 값 :", c, "\n Accuracy :", a)


#  Accuracy : 0.75