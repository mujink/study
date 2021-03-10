import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x1_data = [73.,93.,89.,96.,73.] 
x2_data = [80.,88.,91.,98.,66.]
x3_data = [75.,93.,90.,100.,70.]
y_data = [152.,185.,180.,196.,142.]


x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

fd ={x1:x1_data,x2:x2_data,x3:x3_data,y:y_data}

w1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]), name='weight1')
w2= tf.compat.v1.Variable(tf.compat.v1.random.normal([1]), name='weight2')
w3 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]), name='weight3')
b = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]), name='bias')


hypothesis = x1*w1 + x2*w2 + x3*w3 + b
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
# print(hypothesis)
print(sess.run(hypothesis, fd))

cost = tf.compat.v1.reduce_mean(tf.square(hypothesis-y))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0000452).minimize(cost)


with tf.compat.v1.Session() as sess :
    sess.run(tf.compat.v1.global_variables_initializer())
        
    for step in range(2001):
        _, cost_val, hy_val = sess.run([train, cost, hypothesis], feed_dict=fd)
        if step % 20 ==0:
            print(step,"cost :", cost_val, "\n",hy_val)