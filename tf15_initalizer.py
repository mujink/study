import tensorflow as tf

a = [1,2,3]
tf.compat.v1.initialize_variables(var_list=a, name='init')

tf.compat.v1.initialize_all_variables
tf.compat.v1.initialize_local_variables

tf.compat.v1.constant_initializer
tf.compat.v1.glorot_normal_initializer
tf.compat.v1.glorot_uniform_initializer
tf.compat.v1.identity
tf.compat.v1.ones_initializer
tf.compat.v1.orthogonal_initializer
tf.compat.v1.random_normal_initializernormal
tf.compat.v1.random_uniform_initializer
tf.compat.v1.truncated_normal_initializer
tf.compat.v1.uniform_unit_scaling_initializer
tf.compat.v1.variance_scaling_initializer
tf.compat.v1.zeros_initializer

# fun
tf.compat.v1.global_variables_initializer
# he_normal()
Conv2D(32, (2,2), kernel_initializer="he_normal")

# tf.compat.v1.he_uniform()
# tf.compat.v1.lecun_normal()
# tf.compat.v1.lecun_uniform()
tf.compat.v1.local_variables_initializer
tf.compat.v1.tables_initializer
tf.compat.v1.variables_initializer
