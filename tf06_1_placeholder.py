import tensorflow as tf

tf.compat.v1.set_random_seed(66)

# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])
fd ={x_train:[1,2,3],y_train:[3,5,7]}
W = tf.Variable(tf.random.normal([1]), name='weight')
B = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = x_train * W + B

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.compat.v1.Session() as sess :
    sess.run(tf.compat.v1.global_variables_initializer())
        
    for step in range(401):
        # sess.run(train)
        _, cost_val, w_val, b_val = sess.run([train, cost, W, B], feed_dict=fd)
        if step % 20 ==0:
            # print(step, sess.run(cost), sess.run(W), sess.run(B))
            print(step, cost_val, w_val, b_val)
