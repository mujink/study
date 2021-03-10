import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

x_data = [[1,2],[2,3],[3,1],
          [4,3],[5,3],[6,2]]

y_data = [[0],[0],[0],
          [1],[1],[1]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

fd ={x:x_data,y:y_data}


w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name="weight")
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name="bias")

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
# [실습] 맹그러봐
# cost = tf.compat.v1.reduce_mean(tf.square(hypothesis-y))
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=0.16).minimize(cost)
# train = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-3).minimize(cost)

# cast => hypothesis > 0.5 가 True 면 1, False 면 0 반환
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# predicted와 y가 같은지 비교하고 참이면 1, 거짓이면 0을 출력해서 평균을 구함
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict=fd)
        if step % 200 ==0:
            print(step,"cost :", cost_val)
    
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=fd)
    print("예측 값 :", h, "\n 원래 값 :", c, "\n Accuracy :", a)

