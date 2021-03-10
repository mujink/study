import tensorflow as tf

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)


sess = tf.compat.v1.Session()
# sess = tf.Session()


# a =  tf.placeholder(tf.float32) # The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
# b =  tf.placeholder(tf.float32)
a =  tf.compat.v1.placeholder(tf.float32)
b =  tf.compat.v1.placeholder(tf.float32)
adder_node = a+b
# feed_dict input value
print(sess.run(adder_node, feed_dict={a:3,b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[3,4]}))

add_and_triple = adder_node *3
print(sess.run(add_and_triple, feed_dict={a:4,b:2}))
# print(sess.run(add_and_triple, feed_dict={a:[1,3], b:[3,4]}))

# print(node3)        #Tensor("Add:0", shape=(), dtype=float32)
#                     #tf.Tensor(7.0, shape=(), dtype=float32)
# print('sess.run(node3) :',sess.run(node3))
# print('sess.run(node1,2) :',sess.run(node1),sess.run(node2))
# print('sess.run(node1,2) :',sess.run([node1,node2]))