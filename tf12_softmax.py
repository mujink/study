import tensorflow as tf
import numpy as np
tf.set_random_seed(66)


x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

x =  tf.compat.v1.placeholder(tf.float32, shape=[None,4])
y =  tf.compat.v1.placeholder(tf.float32, shape=[None,3])
fd = {x:x_data, y:y_data}

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,3]), name="weight")
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1,3]), name="bias")

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# predicted와 y가 같은지 비교하고 참이면 1, 거짓이면 0을 출력해서 평균을 구함
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(4200):
        loss_val, _ = sess.run([loss, train], feed_dict=fd)
        if step % 200 ==0:
            print(step,"cost :", loss_val)
    # h = sess.run([hypothesis], feed_dict=fd)#, predicted, accuracy], feed_dict=fd)
    # print("hypothesis :", h)

    a = sess.run(hypothesis, feed_dict={x:[[1,11,7,9]]})
    print("hypothesis :",a, "\n 답", sess.run(tf.argmax(a,1)))