from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
                   
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1,28*28)/255.
x_test = x_test.reshape(-1,28*28)/255.

# (60000, 784)
# (10000, 784)
# (10000, 10)
# (60000, 10)
# print(y_train)
# print(y_test)
x = tf.compat.v1.placeholder(tf.float32, shape=[None,28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])

fd = {x:x_train, y:y_train}
fdt = {x:x_test, y:y_test}

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([28*28,10]), name="weight1")
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10]), name="bias1")
# w1 = tf.Variable(tf.zeros([784, 10]))
# b1 = tf.Variable(tf.zeros([10]))
hypothesis = tf.nn.softmax(tf.matmul(x, w1) + b1)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.5).minimize(loss)



hy = tf.argmax(y,1)
predicted = tf.argmax(hypothesis,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,hy), dtype=tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(501):
        loss_val, p, Acc,_ = sess.run([loss, predicted, accuracy, train], feed_dict=fd)
        if step % 20 ==0:
            y_tr = sess.run(tf.argmax(y_train,1))
            acc = accuracy_score(p, y_tr)
            print(step,"cost :", loss_val, "acc :", acc, "Acc :", Acc)
    loss_val, p, _ = sess.run([loss, hypothesis, train], feed_dict=fdt)
    ps = sess.run(tf.argmax(p,1))
    y_te = sess.run(tf.argmax(y_test,1))
    acc = accuracy_score(ps, y_te)
    print("test cost :", loss_val, "test acc :", acc)


#  Accuracy : 0.75