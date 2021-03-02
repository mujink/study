import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

# print(model.trainable_weights)
print(len(model.weights))
print(len(model.trainable_weights))
"""
print(len(model.weights)) = 8
print(model.weights)
# 지금 들어있는 웨이트 값

[<tf.Variable 'dense/kernel:0' shape=(1, 4) dtype=float32, numpy=
array([[ 0.2177527 , -0.4398648 , -0.675913  , -0.28801954]],
      dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(4, 3) dtype=float32, numpy=
array([[ 0.81372035, -0.48411277, -0.14459759],
       [ 0.32744324, -0.8833189 ,  0.34132516],
       [ 0.82632685, -0.8161093 ,  0.4051715 ],
       [ 0.21299183,  0.8343891 , -0.70027804]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(3, 
2) dtype=float32, numpy=
array([[-0.29462755,  0.94142485],
       [-1.0333468 ,  0.06051934],
       [ 1.0880141 , -0.26922965]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_3/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-0.77552664],
       [ 0.10357869]], dtype=float32)>, <tf.Variable 'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
"""

"""
print(len(model.trainable_weights)) = 8
print(model.trainable_weights)
# 트레인한 값

[<tf.Variable 'dense/kernel:0' shape=(1, 4) dtype=float32, numpy=
array([[-0.18349338,  0.9023683 , -0.47251034,  0.55495393]],
      dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(4, 3) dtype=float32, numpy=
array([[ 0.7232872 ,  0.08969247,  0.04037887],
       [ 0.01615787, -0.7455294 ,  0.265813  ],
       [-0.6004447 ,  0.3255558 ,  0.775082  ],
       [-0.5153791 , -0.6611022 ,  0.8455442 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(3, 
2) dtype=float32, numpy=
array([[ 0.561515  , -1.0204957 ],
       [-0.32097793, -0.7906028 ],
       [-0.7705473 , -0.9211757 ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_3/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-1.0160054],
       [-0.6304322]], dtype=float32)>, <tf.Variable 'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
"""