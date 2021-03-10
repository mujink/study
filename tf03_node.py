import tensorflow as tf

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)


sess = tf.compat.v1.Session()
sess = tf.Session()
print(node3)        #Tensor("Add:0", shape=(), dtype=float32)
                    #tf.Tensor(7.0, shape=(), dtype=float32)
print('sess.run(node3) :',sess.run(node3))
print('sess.run(node1,2) :',sess.run(node1),sess.run(node2))
print('sess.run(node1,2) :',sess.run([node1,node2]))