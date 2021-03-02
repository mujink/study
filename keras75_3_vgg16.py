from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(weights="imagenet", include_top= False, input_shape=(32,32,3))
# include_top이 False여야 인풋쉐이프를 바꿀 수 있음
# vgg16.trainable = True  # 해당 레이어 가중치 동결하지 않음
vgg16.trainable = False # 해당 레이어 가중치 동결
print(vgg16.summary())
print("그냥 가중치의 수:",len(vgg16.weights))           #26
print("동결전 훈련되는 가중치의 수",len(vgg16.trainable_weights)) #0

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))#, activation="softmax"))
model.summary()

print("그냥 가중치의 수:",len(vgg16.weights))           #32
print("동결후 훈련되는 가중치의 수",len(vgg16.trainable_weights)) #6

"""
# vgg16.trainable = True
===========================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
===========================
"""
"""
print(len(vgg16.weights))
print(len(vgg16.trainable_weights))

그냥 가중치의 수: 26
동결후 훈련되는 가중치의 수 26
"""
"""
# vgg16.trainable = False
===========================
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688
===========================
"""
"""
print(len(vgg16.weights))
print(len(vgg16.trainable_weights))

그냥 가중치의 수: 26
동결후 훈련되는 가중치의 수 0
"""
import pandas as pd
pd.set_option('max_colwidth', -1)
layer = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layer, columns= ["Layer Type", "Layer Name", "Layer Trainable"])

print(aaa)