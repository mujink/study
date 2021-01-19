import numpy as np

# data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 2 model

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3 compile, fit

from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Adam(lr=0.1)
# loss :  0.8494184613227844 y_pred :  [[12.966707]]

# optimizer = Adam(lr=0.01)
# loss :  0.009967544116079807 y_pred :  [[11.189517]]

# optimizer = Adam(lr=0.001)
# loss :  7.157154729986459e-11 y_pred :  [[11.000016]]

# optimizer = Adam(lr=0.0001)
# loss :  2.78911720670294e-05 y_pred :  [[10.997059]]
# ============================================================
# optimizer = Adadelta(lr=0.1)
# loss :  0.0032457425259053707 y_pred :  [[11.102851]]

# optimizer = Adadelta(lr=0.01)
# loss :  1.1453451406850945e-05 y_pred :  [[10.993094]]

# optimizer = Adadelta(lr=0.001)
# loss :  12.923612594604492 y_pred :  [[4.5785303]]
# ============================================================
# optimizer = Adamax(lr=0.1)
# loss :  32.24275207519531 y_pred :  [[15.299248]]

# optimizer = Adamax(lr=0.01)
# loss :  1.0700774034227978e-12 y_pred :  [[10.999999]]

# optimizer = Adamax(lr=0.001)
# loss :  1.3356395989205794e-08 y_pred :  [[10.999757]]
# ============================================================

# optimizer = Adagrad(lr=0.1)
# loss :  0.7076201438903809 y_pred :  [[11.1262045]]

# optimizer = Adagrad(lr=0.01)
# loss :  9.321547622676007e-07 y_pred :  [[11.000604]]

# optimizer = Adagrad(lr=0.001)
# loss :  9.459159628022462e-06 y_pred :  [[10.995831]]

# optimizer = Adagrad(lr=0.001)
# loss :  2.1533718609134667e-05 y_pred :  [[10.995809]]
# ============================================================

# optimizer = RMSprop(lr=0.01)
# loss :  2.207491397857666 y_pred :  [[7.825623]]

# optimizer = RMSprop(lr=0.001)
# loss :  0.030038025230169296 y_pred :  [[10.646792]]

# optimizer = RMSprop(lr=0.0001)
# loss :  0.00043499600724317133 y_pred :  [[10.961597]]

# optimizer = RMSprop(lr=0.00001)
# loss :  5.387390046962537e-06 y_pred :  [[10.995292]]

# optimizer = RMSprop(lr=0.000001)
# loss :  6.924454689025879 y_pred :  [[6.2923384]]
# ============================================================

# optimizer = SGD(lr=0.01)
# loss :  nan y_pred :  [[nan]]

# optimizer = SGD(lr=0.001)
# loss :  7.841959813958965e-06 y_pred :  [[10.994494]]

# optimizer = SGD(lr=0.0001)
# loss :  0.0015226805116981268 y_pred :  [[10.952346]]
# ============================================================

# optimizer = Nadam(lr=0.1)
# loss :  12.602835655212402 y_pred :  [[15.236937]]

# optimizer = Nadam(lr=0.01)
# loss :  9.379164112033322e-13 y_pred :  [[11.000004]] @@@@@@@@@@@@@@@@

# optimizer = Nadam(lr=0.001)
# loss :  2.149036591042597e-12 y_pred :  [[11.000001]] 

optimizer = Nadam(lr=0.0001)
# loss :  5.9579133449005894e-06 y_pred :  [[10.995723]]
# ============================================================


model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x,y,epochs=100, batch_size=1)

#4. evaluate
loss, mse = model.evaluate(x,y,batch_size=1)
y_pred = model.predict([11])
print("loss : ", loss, "y_pred : ", y_pred)