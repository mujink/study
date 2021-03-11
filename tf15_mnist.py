from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# (60000, 28, 28)
# (60000,)
# (10000, 28, 28)
# (10000,)

tf.compat.v1.set_random_seed(66)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()                           
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)                          
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()

x_train = x_train.reshape(-1,28*28)/255.
x_test = x_test.reshape(-1,28*28)/255.

# (60000, 784)
# (10000, 784)
# (10000, 10)
# (60000, 10)
# print(y_train)
# print(y_test)
fd = {x:x_train, y:y_train}
fdt = {x:x_test, y:y_test}

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([28*28,64], stddev=0.1), name="weight1")
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64],stddev=0.1), name="bias1")
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64,32],stddev=0.1), name="weight2")
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32],stddev=0.1), name="bias2")
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,10],stddev=0.1), name="weight3")
b3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10],stddev=0.1), name="bias3")
hypothesis = tf.nn.softmax(tf.matmul(layer2, w3) + b3)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss)


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