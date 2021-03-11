from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

                
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1,28*28)/255.
x_test = x_test.reshape(-1,28*28)/255.

x = tf.compat.v1.placeholder(tf.float32, shape=[None,28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])

fd = {x:x_train, y:y_train}
fdt = {x:x_test, y:y_test}

# w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([784,10]), name="weight1")
w1 = tf.compat.v1.get_variable("w1", shape=[784,100],
                                initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100]), name="bias1")
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# layer1 = tf.nn.dropout(layer1, keep_prob=0.3)



# w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,50]), name="weight2")
w2 = tf.compat.v1.get_variable("w2", shape=[100,128],
                                initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([128]), name="bias2")
layer2 = tf.nn.elu(tf.matmul(layer1, w2) + b2)
# layer2 = tf.nn.dropout(layer2, keep_prob=0.3)

w3 = tf.compat.v1.get_variable("w3", shape=[128,64],
                                initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64]), name="bias3")
layer3 = tf.nn.selu(tf.matmul(layer2, w3) + b3)
# layer3 = tf.nn.dropout(layer3, keep_prob=0.3)


w4 = tf.compat.v1.get_variable("w4", shape=[64,10],
                                initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10]), name="bias4")
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)
# hypothesis = tf.nn.dropout(layer4, keep_prob=0.3)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

training_epoch = 3
batch_size = 100
total_batch = int(len(x_train)/batch_size) #60000/100 = 600

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())



for epoch in range(training_epoch): #15 에포
    avg_cost = 0
    for i in range(total_batch): #600 번 돌음
        start = i*batch_size     #100
        end = start + batch_size #200

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}

        hy = tf.argmax(batch_y,1)
        predicted = tf.argmax(hypothesis,1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,hy), dtype=tf.float32))

        c, _, acc = sess.run([loss, train, accuracy], feed_dict=feed_dict)

        avg_cost += c/total_batch #600/600
    print("Epoch :", '%04d' %(epoch + 1),
          "loss = {:.9f}".format(avg_cost),
          "Acc : ",acc)



print("훈련 끝!!")

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.math.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

p = sess.run(hypothesis,feed_dict={x:x_test})
p.reshape(-1,1)
ps = sess.run(tf.argmax(p,1))
y_test = sess.run(tf.argmax(y_test,1))
acc = accuracy_score(ps, y_test)

print("Acc :", sess.run(accuracy, feed_dict=fdt), "acc:", acc)

    # p = sess.run([hypothesis], feed_dict=fdt)
    # p.reshape(-1,1)
    # ps = sess.run(tf.argmax(p,1))
    # y_te = sess.run(tf.argmax(y_test,1))
    # acc = accuracy_score(ps, y_te)
    # print( "Accuracy :", acc)