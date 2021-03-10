import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

dataset = np.loadtxt('../data/csv/data-01-test-score.csv', delimiter=',')

x_train = dataset[:,:-1]
y_train = dataset[:,-1]
y_train = y_train.reshape(-1,1)
print(x_train.shape)
print(y_train.shape)
# (25, 3)
# (25,)

print(x_train)
print(y_train)

tf.compat.v1.set_random_seed(66)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

fd ={x:x_train,y:y_train}


w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]), name="weight")
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name="bias")

# hypothesis =  x*w+b
hypothesis =  tf.matmul(x, w) + b

# [실습] 맹그러봐
cost = tf.compat.v1.reduce_mean(tf.square(hypothesis-y))
# train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=0.16).minimize(cost)
train = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-3).minimize(cost)

with tf.compat.v1.Session() as sess :
    sess.run(tf.compat.v1.global_variables_initializer())
        
    for step in range(15001):
        _, cost_val, hy_val = sess.run([train, cost, hypothesis], feed_dict=fd)
        if step % 20 ==0:
            print(step,"cost :", cost_val)#, "\n hy :",hy_val)
    print("x : [73,80,75]",  "Predict : ",sess.run(hypothesis, feed_dict={x:[[73,80,75]]}))
    print("x : [93,88,93]",  "Predict : ",sess.run(hypothesis, feed_dict={x:[[93,88,93]]}))
    print("x : [89,91,90]",  "Predict : ",sess.run(hypothesis, feed_dict={x:[[89,91,90]]}))
    print("x : [96,98,100]",  "Predict : ",sess.run(hypothesis, feed_dict={x:[[96,98,100]]}))
    print("x : [73,66,142]",  "Predict : ",sess.run(hypothesis, feed_dict={x:[[73,66,142]]}))

# x : [73,80,75] Predict :  [[152.5447]]
# x : [93,88,93] Predict :  [[184.86891]]
# x : [89,91,90] Predict :  [[181.58482]]
# x : [96,98,100] Predict :  [[199.46022]]
# x : [73,66,142] Predict :  [[222.72871]]