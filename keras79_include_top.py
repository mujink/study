from tensorflow.keras.applications import VGG16


model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
# model = VGG16()

model.trainable=False
model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

# model = VGG16(weights="imagenet", include_top= False, input_shape=(224,224,3))
# =================================================================
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688
# _________________________________________________________________
# 26
# 0

# model = VGG16()
# =================================================================
# Total params: 138,357,544
# Trainable params: 0
# Non-trainable params: 138,357,544
# _________________________________________________________________
# 32
# 0

# model = VGG16(weights="imagenet", include_top=True, input_shape=(224,224,3))
# =================================================================
# Total params: 138,357,544
# Trainable params: 0
# Non-trainable params: 138,357,544
# _________________________________________________________________
# 32
# 0



# Model: "vgg16"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 224, 224, 3)]     0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
# =================================================================
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688
# _________________________________________________________________


# Model: "vgg16"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 224, 224, 3)]     0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
# _________________________________________________________________
# _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              102764544
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 0
# Non-trainable params: 138,357,544
# _________________________________________________________________