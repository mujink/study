import tensorflow as tf

tf.compat.v1.set_random_seed(66)

x_data = [[73,51,65],
          [92,98,11],
          [89,31,33],
          [99,33,100],
          [17,66,79]]

y_data = [[152],
          [185],
          [180],
          [205],
          [142]]

x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None,1])

fd ={x:x_data,y:y_data}


w = tf.Variable(tf.random_normal([3,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# hypothesis =  x*w+b
hypothesis =  tf.matmul(x, w) + b

# [실습] 맹그러봐
cost = tf.compat.v1.reduce_mean(tf.square(hypothesis-y))
train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=0.1).minimize(cost)
# train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.compat.v1.Session() as sess :
    sess.run(tf.compat.v1.global_variables_initializer())
        
    for step in range(10000):
        _, cost_val, hy_val = sess.run([train, cost, hypothesis], feed_dict=fd)
        if step % 20 ==0:
            print(step,"cost :", cost_val, "\n hy :",hy_val)
